from models.IGM.unet import UNet
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import cfg
from src.lap_solvers.hungarian import hungarian
import random
from src.lap_solvers.sinkhorn import Sinkhorn


class ResCls(nn.Module):
    def __init__(self, n, intro, unit, outro):
        super().__init__()
        self.verse = nn.ModuleList([nn.BatchNorm1d(unit) for _ in range(n)])
        self.chorus = nn.ModuleList([nn.Conv1d(unit, unit, 1) for _ in range(n)])
        self.intro = nn.Conv1d(intro, unit, 1)
        self.outro = nn.Conv1d(unit, outro, 1)

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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(True)  # UNet(3, 2)
        # self.unet.load_state_dict(torch.load("unet_carvana_scale0.5_epoch1.pth"))
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512 * 2) * 4
        self.cls = ResCls(3, feature_lat, 2048, 64)
        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON
        )

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
        return F_src, F_tgt

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = list(self.encode(src)), list(self.encode(tgt))
        F_src, F_tgt = self.halo(feat_srcs, feat_tgts, P_src, P_tgt)

        y_src, y_tgt = self.cls(F_src), self.cls(F_tgt)
        # sim = self.attn(y_src, y_tgt, P_src, P_tgt, ns_src, ns_tgt)
        # sim = self.projection(y_src), self.projection(y_tgt)

        sim = torch.einsum(
            "bci,bcj->bij",
            y_src - y_src.mean(-1, keepdim=True),
            y_tgt - y_tgt.mean(-1, keepdim=True)
        )
        data_dict['ds_mat'] = self.sinkhorn(sim, ns_src, ns_tgt, dummy_row=True)
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        return data_dict
