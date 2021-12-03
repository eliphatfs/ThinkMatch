from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import cfg
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.sinkhorn import Sinkhorn
from extra.pointnetpp import p2_smaller
from models.BBGM.sconv_archs import SiameseSConvOnNodes


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


class HALO(nn.Module):
    def __init__(self, sinkhorn, n_emb, rescale):
        super().__init__()
        self.rescale = rescale
        self.sinkhorn = sinkhorn
        self.projection_g = ResCls(1, 1024, 256, 128)
        self.pp2 = p2_smaller.MultiScalePropagation(n_emb, 128, n_emb)
        self.attn = p2_smaller.HALOAttention(sinkhorn, ResCls(0, n_emb, n_emb * 2 + 4, n_emb), n_emb, 36)

    def prepare_p(self, y_src, P_src, n_src):
        resc = P_src.new_tensor(self.rescale)
        P_src = P_src / resc
        P_src = P_src.transpose(1, 2)
        key_mask_src = torch.arange(y_src.shape[-1], device=n_src.device).expand(len(y_src), y_src.shape[-1]) < n_src.unsqueeze(-1)
        P_src = torch.cat((P_src, torch.zeros_like(P_src[:, :1])), 1)
        cat = torch.cat((P_src, y_src), 1) * key_mask_src.unsqueeze(1)
        return cat

    def forward(self, x_src, x_tgt, P_src, P_tgt, g_src, g_tgt, n_src, n_tgt):
        g_src, g_tgt = self.projection_g(g_src), self.projection_g(g_tgt)
        g_src, g_tgt = F.normalize(g_src, 1), F.normalize(g_tgt, 1)
        x_src, x_tgt = self.prepare_p(x_src, P_src, n_src), self.prepare_p(x_tgt, P_tgt, n_tgt)
        x_src, x_tgt = self.pp2(x_src, g_src), self.pp2(x_tgt, g_tgt)
        return self.attn(x_src, x_tgt, n_src, n_tgt)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512 * 2)
        self.pix2pt_proj = ResCls(1, feature_lat, 512, 384)
        self.tau = 5  # cfg.IGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.IGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.IGM.SK_EPSILON
        )
        self.final_proj = ResCls(1, 384, 384, 36)
        self.halo_1 = HALO(self.sinkhorn, 384, self.rescale)
        self.halo_2 = HALO(self.sinkhorn, 384, self.rescale)
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

    def merge_feat(self, feat_srcs, feat_tgts, P_src, P_tgt):
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

    def points(self, y_src, P_src, n_src, cls, g):
        resc = P_src.new_tensor(self.rescale)
        P_src = P_src / resc
        P_src = P_src.transpose(1, 2)
        key_mask_src = torch.arange(y_src.shape[-1], device=n_src.device).expand(len(y_src), y_src.shape[-1]) < n_src.unsqueeze(-1)
        P_src = torch.cat((P_src, torch.zeros_like(P_src[:, :1])), 1)
        return self.pn(torch.cat((P_src, y_src), 1) * key_mask_src.unsqueeze(1), cls, g)[..., :y_src.shape[-1]]

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']
        self.sinkhorn.batched_operation = not self.training

        feat_srcs, feat_tgts = [], []
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        if self.training:
            P_src = P_src + torch.randn_like(P_src)
            P_tgt = P_tgt + torch.randn_like(P_tgt)
            P_src = P_src + torch.rand_like(P_src[..., :1]) * 20 - 10
            P_tgt = P_tgt + torch.rand_like(P_tgt[..., :1]) * 20 - 10
        F_src, F_tgt, g_src, g_tgt = self.merge_feat(feat_srcs, feat_tgts, P_src, P_tgt)

        y_src, y_tgt = self.pix2pt_proj(F_src), self.pix2pt_proj(F_tgt)
        y_src, y_tgt = F.normalize(y_src, dim=1), F.normalize(y_tgt, dim=1)
        y_src, y_tgt = self.halo_1(y_src, y_tgt, P_src, P_tgt, g_src, g_tgt, ns_src, ns_tgt)
        y_src, y_tgt = self.halo_2(y_src, y_tgt, P_src, P_tgt, g_src, g_tgt, ns_src, ns_tgt)
        folding_src, folding_tgt = self.final_proj(y_src), self.final_proj(y_tgt)
        sim = torch.einsum(
            'bci,bcj->bij',
            folding_src,
            folding_tgt
        )
        data_dict['ds_mat'] = self.sinkhorn(sim, ns_src, ns_tgt, dummy_row=True)
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        return data_dict
