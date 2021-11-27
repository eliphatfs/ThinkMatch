from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import cfg
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.sinkhorn import Sinkhorn
from extra.optimal_transport import SinkhornDistance
import numpy
import sys


numpy.set_printoptions(formatter={'float': lambda x: "%.2f" % x if abs(x) > 0.01 else "----"})


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
        self.resnet = resnet34(True)  # UNet(3, 2)
        # self.unet.load_state_dict(torch.load("unet_carvana_scale0.5_epoch1.pth"))
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512 * 2)
        self.cls = nn.Identity()  # ResCls(1, feature_lat, 2048, 2048)
        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pf = torch.nn.Sequential(
            torch.nn.Conv1d(4, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 256, 1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )
        self.pn = torch.nn.Sequential(
            torch.nn.Conv1d(feature_lat + 256, 2048, 1),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Conv1d(2048, 3072, 1),
            torch.nn.BatchNorm1d(3072),
            torch.nn.ReLU(),
            torch.nn.Conv1d(3072, 3072, 1),
            torch.nn.BatchNorm1d(3072),
            torch.nn.ReLU(),
        )
        self.pe = torch.nn.Sequential(
            torch.nn.Conv1d(3072 + 256, 3072, 1),
            torch.nn.BatchNorm1d(3072),
            torch.nn.ReLU(),
            torch.nn.Conv1d(3072, 1024, 1),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Conv1d(1024, 256, 1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 2, 1)
        )
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON
        )
        self.ot = SinkhornDistance(0.02, 10, 'mean')
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
        return F_src, F_tgt

    def points(self, y_src, y_tgt, P_src, P_tgt, n_src, n_tgt):
        resc = P_src.new_tensor(self.rescale)
        P_src, P_tgt = P_src / resc, P_tgt / resc
        P_src, P_tgt = P_src.transpose(1, 2), P_tgt.transpose(1, 2)
        if self.training:
            P_src = P_src + torch.randn_like(P_src)[..., :1] * 0.1
            P_tgt = P_tgt + torch.randn_like(P_tgt)[..., :1] * 0.1
        key_mask_src = torch.arange(y_src.shape[-1], device=n_src.device).expand(len(y_src), y_src.shape[-1]) < n_src.unsqueeze(-1)
        key_mask_tgt = torch.arange(y_tgt.shape[-1], device=n_tgt.device).expand(len(y_tgt), y_tgt.shape[-1]) < n_tgt.unsqueeze(-1)
        key_mask_cat = torch.cat((key_mask_src, key_mask_tgt), -1).unsqueeze(1)
        P_src = torch.cat((P_src, torch.zeros_like(P_src)), 1)
        P_tgt = torch.cat((P_tgt, torch.ones_like(P_tgt)), 1)
        pcd = self.pf(torch.cat((P_src, P_tgt), -1))
        y_cat = torch.cat((y_src, y_tgt), -1)
        pcc = self.pn(torch.cat((pcd, y_cat), 1) * key_mask_cat).max(-1, keepdim=True)[0]
        pcc_b = pcc.expand(pcc.shape[0], pcc.shape[1], y_cat.shape[-1])
        return self.pe(torch.cat((pcc_b, pcd), 1))[..., :y_src.shape[-1]]

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = [], []
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        F_src, F_tgt = self.halo(feat_srcs, feat_tgts, P_src, P_tgt)

        y_src, y_tgt = self.cls(F_src), self.cls(F_tgt)
        folding_src = self.points(y_src, y_tgt, P_src, P_tgt, ns_src, ns_tgt).transpose(1, 2)
        folding_tgt = self.points(y_tgt, y_src, P_tgt, P_src, ns_tgt, ns_src).transpose(1, 2)
        sim = torch.zeros(y_src.shape[0], y_src.shape[-1], y_tgt.shape[-1]).to(y_src)
        for b in range(len(y_src)):
            sim[b, :ns_src[b], :ns_tgt[b]] = self.ot(
                folding_src[b: b + 1, :ns_src[b]],
                folding_tgt[b: b + 1, :ns_tgt[b]],
            )[1].squeeze(0)
        if torch.rand(1) < 0.005:
            print("S = ", sim[0].detach().cpu().numpy(), file=sys.stderr)
            print("G = ", data_dict['gt_perm_mat'][0].detach().cpu().numpy(), file=sys.stderr)
            print("Ps = ", folding_src[0].detach().cpu().numpy(), file=sys.stderr)
            print("Pt = ", folding_tgt[0].detach().cpu().numpy(), file=sys.stderr)
        data_dict['ds_mat'] = sim
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        return data_dict
