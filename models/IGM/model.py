from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import cfg
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.sinkhorn import Sinkhorn


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
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512)
        self.cls = ResCls(2, feature_lat, 2048, 1024)
        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pf = ResCls(1, 3, 128, 256)
        self.pn = ResCls(1, (1024 + 256) * 2, 2048, 2048, 2)
        self.pe = ResCls(2, 3072 + 1024 + 256, 2048, 1, 2)
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

    def align(self, feat_srcs, feat_tgts, P_src, P_tgt):
        U_src = torch.cat([
            my_align(feat_src, P_src, self.rescale) for feat_src in feat_srcs
        ], 1)
        U_tgt = torch.cat([
            my_align(feat_tgt, P_tgt, self.rescale) for feat_tgt in feat_tgts
        ], 1)
        return U_src, U_tgt

    def points(self, y_src, y_tgt, P_src, P_tgt, n_src, n_tgt):
        resc = P_src.new_tensor(self.rescale)
        P_src, P_tgt = P_src / resc, P_tgt / resc
        P_src, P_tgt = P_src.transpose(1, 2), P_tgt.transpose(1, 2)
        if self.training:
            P_src = P_src + torch.randn_like(P_src)[..., :1] * 0.1
            P_tgt = P_tgt + torch.randn_like(P_tgt)[..., :1] * 0.1
        key_mask_src = torch.arange(y_src.shape[-1], device=n_src.device).expand(len(y_src), y_src.shape[-1]) < n_src.unsqueeze(-1)
        key_mask_tgt = torch.arange(y_tgt.shape[-1], device=n_tgt.device).expand(len(y_tgt), y_tgt.shape[-1]) < n_tgt.unsqueeze(-1)
        P_src = self.pf(torch.cat((P_src, torch.zeros_like(P_src[:, :1])), 1))
        P_tgt = self.pf(torch.cat((P_tgt, torch.ones_like(P_tgt[:, :1])), 1))
        yc_src = torch.cat((P_src, y_src), 1)
        yc_tgt = torch.cat((P_tgt, y_tgt), 1)
        
        e_src = torch.cat((yc_src, torch.zeros_like(yc_src)), 1).unsqueeze(-1)
        e_tgt = torch.cat((torch.zeros_like(yc_tgt), yc_tgt), 1).unsqueeze(-2)
        pcd = e_src + e_tgt
        mask = key_mask_src.unsqueeze(-1) & key_mask_tgt.unsqueeze(-2)

        pcc = (self.pn(pcd) * mask.unsqueeze(1)).flatten(2).max(-1)[0].unsqueeze(-1).unsqueeze(-1)
        pcc_b = pcc.expand(pcc.shape[0], pcc.shape[1], pcd.shape[-2], pcd.shape[-1])
        return self.pe(torch.cat((pcc_b, pcd), 1)).squeeze(1)

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = [], []
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        F_src, F_tgt = self.align(feat_srcs, feat_tgts, P_src, P_tgt)

        y_src, y_tgt = self.cls(F_src), self.cls(F_tgt)

        sim1 = self.points(y_src, y_tgt, P_src, P_tgt, ns_src, ns_tgt)
        sim2 = self.points(y_tgt, y_src, P_tgt, P_src, ns_tgt, ns_src)
        
        data_dict['ds_mat'] = self.sinkhorn(sim1 + sim2, ns_src, ns_tgt, dummy_row=True)
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        return data_dict
