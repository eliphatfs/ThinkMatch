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
        return torch.relu(self.outro(x))


def my_align(raw_feature, P, ori_size: tuple):
    return F.grid_sample(
        raw_feature,
        2 * P.unsqueeze(-2) / ori_size[0] - 1,
        'bilinear',
        'border',
        align_corners=False
    ).squeeze(-1)


def my_cdist2(src, dst):  # BPC, BRC -> BPR
    # BP1C, B1RC -> BPRC -> BPR
    return (((src.unsqueeze(-2).contiguous() - dst.unsqueeze(-3).contiguous()) ** 2).sum(-1) + 1e-8) ** 0.5


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)  # UNet(3, 2)
        # self.unet.load_state_dict(torch.load("unet_carvana_scale0.5_epoch1.pth"))
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512 * 2)
        self.cls = ResCls(2, feature_lat, 2048, 32)
        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pf = torch.nn.Sequential(
            torch.nn.Conv1d(4, 256, 1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 128, 1)
        )
        self.pn = torch.nn.Sequential(
            torch.nn.Conv1d(feature_lat + 128, 2048, 1),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Conv1d(2048, 1536, 1),
            torch.nn.BatchNorm1d(1536),
            torch.nn.ReLU(),
            torch.nn.Conv1d(1536, 1536, 1),
            torch.nn.BatchNorm1d(1536),
            torch.nn.ReLU(),
        )
        self.pe = torch.nn.Sequential(
            torch.nn.Conv1d(1536 + 128, 4096, 1),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(),
            torch.nn.Conv1d(4096, 256, 1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 32, 1)
        )
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON
        )
        self.ot = SinkhornDistance(0.01, 10, 'mean')
        self.gater = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 1024, 1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 1, 1),
            torch.nn.Flatten(2),
            torch.nn.Sigmoid()
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
        return F_src, F_tgt

    def points(self, y_src, y_tgt, P_src, P_tgt, n_src, n_tgt, ext=32):
        resc = P_src.new_tensor(self.rescale)
        P_src, P_tgt = P_src / resc, P_tgt / resc
        P_src, P_tgt = P_src.transpose(1, 2), P_tgt.transpose(1, 2)
        # if self.training:
        #     P_src = P_src + torch.randn_like(P_src)[..., :1] * 0.1
        #     P_tgt = P_tgt + torch.randn_like(P_tgt)[..., :1] * 0.1
        key_mask_src = torch.arange(y_src.shape[-1], device=n_src.device).expand(len(y_src), y_src.shape[-1]) < n_src.unsqueeze(-1)
        key_mask_tgt = torch.arange(y_tgt.shape[-1], device=n_tgt.device).expand(len(y_tgt), y_tgt.shape[-1]) < n_tgt.unsqueeze(-1)
        key_mask_cat = torch.cat((key_mask_src, key_mask_tgt), -1).unsqueeze(1)
        P_src = torch.cat((P_src, torch.zeros_like(P_src)), 1)
        P_tgt = torch.cat((P_tgt, torch.ones_like(P_tgt)), 1)
        pcd = self.pf(torch.cat((P_src, P_tgt), -1))
        y_cat = torch.cat((y_src, y_tgt), -1)
        pcc = self.pn(torch.cat((pcd, y_cat), 1) * key_mask_cat).max(-1, keepdim=True)[0]
        pcc_b = pcc.expand(pcc.shape[0], pcc.shape[1], y_cat.shape[-1])
        return self.pe(torch.cat((pcc_b, pcd), 1))[..., ext: y_src.shape[-1]]

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = [], []
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        resc = P_src.new_tensor(self.rescale)
        rand_src, rand_tgt = torch.rand(len(P_src), 32, 2).to(P_src), torch.rand(len(P_tgt), 32, 2).to(P_tgt)
        P_src, P_tgt = torch.cat((rand_src * resc, P_src), 1), torch.cat((rand_tgt * resc, P_tgt), 1)
        y_src, y_tgt = self.halo(feat_srcs, feat_tgts, P_src, P_tgt)
        gate_src, gate_tgt = self.gater(feat_srcs[-1]), self.gater(feat_tgts[-1])
        e2_src, e2_tgt = self.cls(y_src)[..., 32:], self.cls(y_tgt)[..., 32:]
        e1_src = self.points(y_src, y_tgt, P_src, P_tgt, 32 + ns_src, 32 + ns_tgt)
        e1_tgt = self.points(y_tgt, y_src, P_tgt, P_src, 32 + ns_tgt, 32 + ns_src)
        sim = torch.einsum(
            "bci,bcj->bij",
            e1_src * gate_src + e2_src * (1 - gate_src),
            e1_tgt * gate_tgt + e2_tgt * (1 - gate_tgt)
            # y_src - y_src.mean(-1, keepdim=True),
            # y_tgt - y_tgt.mean(-1, keepdim=True)
        )
        if torch.rand(1) < 0.01:
            print("S", gate_src.flatten().detach().cpu().numpy(), file=sys.stderr)
            print("T", gate_tgt.flatten().detach().cpu().numpy(), file=sys.stderr)
        data_dict['ds_mat'] = self.sinkhorn(sim)
        try:
            data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        except Exception:
            import traceback; traceback.print_exc()
            import pdb; pdb.set_trace()
        return data_dict
