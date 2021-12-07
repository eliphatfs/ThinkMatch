from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import cfg
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.sinkhorn import Sinkhorn
from extra.pointnetpp import p2_manifold
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
        self.pix2pt_proj = ResCls(1, feature_lat, 784, 256)
        self.pix2cl_proj = ResCls(1, 1024, 512, 128)
        self.fold_1 = ResCls(1, 512 * 2 + 2, 512, 2)
        self.fold_2 = ResCls(1, 512 * 2 + 2, 512, 3)
        # self.edge_proj = ResCls(2, feature_lat * 3 - 512, 1024, 1)
        self.tau = cfg.IGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pn = p2_manifold.get_model(256, 128)
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

    def points(self, y_src, P_src, TP_src, g):
        resc = P_src.new_tensor(self.rescale)
        TP_src = TP_src / resc
        P_src = P_src.transpose(1, 2)
        TP_src = TP_src.transpose(1, 2)
        if self.training:
            P_src = P_src + torch.rand_like(P_src[..., :1]) * 0.2 - 0.1
            TP_src = TP_src + torch.rand_like(TP_src[..., :1]) * 0.2 - 0.1
        TP_src = torch.cat((TP_src, torch.zeros_like(TP_src[:, :1])), 1)
        return self.pn(torch.cat((P_src, y_src), 1), g, TP_src)

    def fold(self, g):
        grid = torch.stack(torch.meshgrid([torch.linspace(0, 1, 6)] * 2), 0).to(g).reshape(1, 2, -1)
        grid = grid.repeat(len(g), 1, 1)
        grid = grid + torch.randn_like(grid) * 0.01
        g = g.repeat(1, 1, grid.shape[-1])
        f1 = grid + self.fold_1(torch.cat([g, grid], 1))
        f2 = torch.cat([grid, torch.ones_like(grid[:, :1])], 1) + self.fold_2(torch.cat([g, f1], 1))
        return f2.transpose(1, 2)

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = [], []
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        F_src, F_tgt, g_src, g_tgt = self.halo(feat_srcs, feat_tgts, P_src, P_tgt)
        
        resc = P_src.new_tensor(self.rescale)
        gf_src, gf_tgt = self.fold(g_src), self.fold(g_tgt)
        if torch.rand([]) < 0.01:
            print(gf_src[0].transpose(1, 2))
        samp_src = gf_src[..., :2] * resc
        samp_tgt = gf_tgt[..., :2] * resc

        F_src, F_tgt, g_src, g_tgt = self.halo(feat_srcs, feat_tgts, samp_src, samp_tgt)

        y_src, y_tgt = self.pix2pt_proj(F_src), self.pix2pt_proj(F_tgt)

        g_src, g_tgt = self.pix2cl_proj(g_src), self.pix2cl_proj(g_tgt)
        y_src, y_tgt = F.normalize(y_src, dim=1), F.normalize(y_tgt, dim=1)
        g_src, g_tgt = F.normalize(g_src, dim=1), F.normalize(g_tgt, dim=1)
        folding_src = self.points(y_src, gf_src, P_src, g_src)
        folding_tgt = self.points(y_tgt, gf_tgt, P_tgt, g_tgt)

        sim = torch.einsum(
            'bci,bcj->bij',
            folding_src,
            folding_tgt
        )
        data_dict['ds_mat'] = self.sinkhorn(sim, ns_src, ns_tgt, dummy_row=True)
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        return data_dict
