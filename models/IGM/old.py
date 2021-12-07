from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import cfg
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.sinkhorn import Sinkhorn
from extra.pointnetpp import p2_smaller
from models.BBGM.sconv_archs import SiameseSConvOnNodes
from src.loss_func import PermutationLoss


loss_fn = PermutationLoss()


class ResCls(nn.Module):
    def __init__(self, n, intro, unit, outro, ndim=1):
        super().__init__()
        BN = [nn.BatchNorm1d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][ndim]
        CN = [lambda x, y, _: nn.Linear(x, y), nn.Conv1d, nn.Conv2d, nn.Conv3d][ndim]
        self.verse = nn.ModuleList([BN(unit) for _ in range(n)])
        self.chorus = nn.ModuleList([CN(unit, unit, 1) for _ in range(n)])
        self.intro = CN(intro, unit, 1)
        self.outro = CN(unit, outro, 1)

    def forward(self, x):
        x = self.intro(x)
        for chorus, verse in zip(self.chorus, self.verse):
            d = torch.relu(verse(x))
            d = chorus(d)
            x = x + d
        return self.outro(x)


def my_align(raw_feature, P, ori_size: tuple):
    return F.grid_sample(
        raw_feature,
        2 * P.unsqueeze(-2) / ori_size[0] - 1,
        'bilinear',
        'border',
        align_corners=False
    ).squeeze(-1)


def batch_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


def unbatch_features(orig, embeddings, num_vertices):
    res = torch.zeros_like(orig)
    cum = 0
    for embedding, num_v in zip(res, num_vertices):
        embedding[:, :num_v] = embeddings[cum: cum + num_v].t()
        cum = cum + num_v
    return res


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)  # UNet(3, 2)
        # self.unet.load_state_dict(torch.load("unet_carvana_scale0.5_epoch1.pth"))
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512 * 2)
        # self.sconv = SiameseSConvOnNodes(48)
        self.pix2pt_proj = ResCls(1, feature_lat, 512, 48)
        self.pix2cl_proj = ResCls(1, 1024, 512, 128)
        # self.edge_proj = ResCls(2, feature_lat * 3 - 512, 1024, 1)
        self.tau = cfg.IGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pn = p2_smaller.get_model(48, 128)
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.IGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.IGM.SK_EPSILON
        )
        self.backbone_params = list(self.resnet.parameters())

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        r = self.resnet
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        yield x
        x = r.maxpool(x)

        x = r.layer1(x)
        yield x
        x = r.layer2(x)
        yield x
        x = r.layer3(x)
        yield x
        x = r.layer4(x)
        yield x
        x = r.avgpool(x)
        yield x

    def halo(self, feat_srcs, feat_tgts, P_src, P_tgt):
        U_src = torch.cat([
            my_align(feat_src, P_src, self.rescale) for feat_src in feat_srcs
        ], 1)
        U_tgt = torch.cat([
            my_align(feat_tgt, P_tgt, self.rescale) for feat_tgt in feat_tgts
        ], 1)
        glob_src = feat_srcs[-1].flatten(1).unsqueeze(-1)
        glob_tgt = feat_tgts[-1].flatten(1).unsqueeze(-1)
        F_src = torch.cat([
            U_src,
            glob_tgt.expand(*glob_tgt.shape[:-1], U_src.shape[-1])
        ], 1)
        F_tgt = torch.cat([
            U_tgt,
            glob_src.expand(*glob_src.shape[:-1], U_tgt.shape[-1])
        ], 1)
        ghalo_src = torch.cat((glob_src, glob_tgt), 1)
        ghalo_tgt = torch.cat((glob_tgt, glob_src), 1)
        return F_src, F_tgt, ghalo_src, ghalo_tgt

    def edge_activations(self, feats, F, P, n):
        # F: BCN
        # P: BN2
        # n: B
        ep = ((P.unsqueeze(-2) + P.unsqueeze(-3)) / 2).flatten(1, 2)  # B N^2 2
        L = (
            torch.cat([F, torch.zeros_like(F)], 1).unsqueeze(-1) +
            torch.cat([torch.zeros_like(F), F], 1).unsqueeze(-2)
        ).flatten(2)  # B2CN^2
        E = torch.cat([
            my_align(feat, ep, self.rescale) for feat in feats
        ], 1)  # BCN^2
        CE = torch.cat([L, E], 1)
        mask = torch.arange(F.shape[-1], device=n.device).expand(len(F), F.shape[-1]) < n.unsqueeze(-1)
        # BN
        mask = mask.unsqueeze(-2) & mask.unsqueeze(-1)
        return (torch.sigmoid(self.edge_proj(CE)) * mask.flatten(1).unsqueeze(1)).reshape(-1, F.shape[-1], F.shape[-1])

    def points(self, y_src, P_src, n_src, cls, g):
        resc = P_src.new_tensor(self.rescale)
        P_src = P_src / resc
        P_src = P_src.transpose(1, 2)
        if self.training:
            P_src = P_src + torch.rand_like(P_src[..., :1]) * 0.2 - 0.1
        # P_src = (P_src - P_src.min(-1, keepdim=True)[0]) / (P_src.max(-1, keepdim=True)[0] - P_src.min(-1, keepdim=True)[0] + 1e-7)
        key_mask_src = torch.arange(y_src.shape[-1], device=n_src.device).expand(len(y_src), y_src.shape[-1]) < n_src.unsqueeze(-1)
        P_src = torch.cat((P_src, torch.zeros_like(P_src[:, :1])), 1)
        return self.pn(torch.cat((P_src, y_src), 1) * key_mask_src.unsqueeze(1), cls, g)[..., :y_src.shape[-1]]

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = [], []
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        if self.training:
            P_src = P_src + torch.rand_like(P_src) * 2 - 1
            P_tgt = P_tgt + torch.rand_like(P_tgt) * 2 - 1
        F_src, F_tgt, g_src, g_tgt = self.halo(feat_srcs, feat_tgts, P_src, P_tgt)

        # ea_src = self.edge_activations(feat_srcs, F_src, P_src, ns_src)
        # ea_tgt = self.edge_activations(feat_tgts, F_tgt, P_tgt, ns_tgt)

        y_src, y_tgt = self.pix2pt_proj(F_src), self.pix2pt_proj(F_tgt)

        g_src, g_tgt = self.pix2cl_proj(g_src), self.pix2cl_proj(g_tgt)
        y_src, y_tgt = F.normalize(y_src, dim=1), F.normalize(y_tgt, dim=1)
        g_src, g_tgt = F.normalize(g_src, dim=1), F.normalize(g_tgt, dim=1)
        folding_src = self.points(y_src, P_src, ns_src, data_dict['cls'][0], g_src)
        folding_tgt = self.points(y_tgt, P_tgt, ns_tgt, data_dict['cls'][1], g_tgt)

        sim = torch.einsum(
            'bci,bcj->bij',
            folding_src,
            folding_tgt
        )
        data_dict['ds_mat'] = self.sinkhorn(sim, ns_src, ns_tgt, dummy_row=True)
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        return data_dict