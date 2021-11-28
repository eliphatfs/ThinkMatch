from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import cfg
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.sinkhorn import Sinkhorn
from extra.pointnetpp import p2_smaller


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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)  # UNet(3, 2)
        # self.unet.load_state_dict(torch.load("unet_carvana_scale0.5_epoch1.pth"))
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512 * 2)
        self.cls = ResCls(0, feature_lat, 2048, 1024)
        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pn = p2_smaller.get_model()
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON
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

    def points(self, y_src, y_tgt, P_src, P_tgt, n_src, n_tgt):
        resc = P_src.new_tensor(self.rescale)
        P_src, P_tgt = P_src / resc, P_tgt / resc
        P_src, P_tgt = P_src.transpose(1, 2), P_tgt.transpose(1, 2)
        if self.training:
            P_src = P_src + torch.rand_like(P_src)[..., :1] * 0.2 - 0.1
            P_tgt = P_tgt + torch.rand_like(P_tgt)[..., :1] * 0.2 - 0.1
        key_mask_src = torch.arange(y_src.shape[-1], device=n_src.device).expand(len(y_src), y_src.shape[-1]) < n_src.unsqueeze(-1)
        key_mask_tgt = torch.arange(y_tgt.shape[-1], device=n_tgt.device).expand(len(y_tgt), y_tgt.shape[-1]) < n_tgt.unsqueeze(-1)
        key_mask_cat = torch.cat((key_mask_src, key_mask_tgt), -1).unsqueeze(1)
        P_src = torch.cat((P_src, torch.zeros_like(P_src[:, :1])), 1)
        P_tgt = torch.cat((P_tgt, torch.ones_like(P_tgt[:, :1])), 1)
        pcd = torch.cat((P_src, P_tgt), -1)
        y_cat = torch.cat((y_src, y_tgt), -1)
        return self.pn(torch.cat((pcd, y_cat), 1) * key_mask_cat)[..., :y_src.shape[-1]]

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = [], []
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        resc = P_src.new_tensor(self.rescale)
        rand_src, rand_tgt = torch.rand(len(P_src), 64, 2).to(P_src), torch.rand(len(P_tgt), 64, 2).to(P_tgt)
        P_src, P_tgt = torch.cat((rand_src * resc, P_src), 1), torch.cat((rand_tgt * resc, P_tgt), 1)
        F_src, F_tgt = self.halo(feat_srcs, feat_tgts, P_src, P_tgt)

        y_src, y_tgt = self.cls(F_src), self.cls(F_tgt)
        folding_src = self.points(y_src, y_tgt, P_src, P_tgt, 64 + ns_src, 64 + ns_tgt)[..., 64:]
        folding_tgt = self.points(y_tgt, y_src, P_tgt, P_src, 64 + ns_tgt, 64 + ns_src)[..., 64:]

        sim = torch.einsum(
            "bxi,bxj->bij",
            folding_src, folding_tgt
        )
        data_dict['ds_mat'] = self.sinkhorn(sim, ns_src, ns_tgt, dummy_row=True)
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        return data_dict
