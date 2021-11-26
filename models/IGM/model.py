from models.IGM.unet import UNet
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.feature_align import feature_align
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(True)  # UNet(3, 2)
        # self.unet.load_state_dict(torch.load("unet_carvana_scale0.5_epoch1.pth"))
        self.cls = ResCls(3, 64 + 2048 * 2, 1536, 32)
        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        r = self.resnet
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        x = r.maxpool(x)

        x = r.layer1(x)
        x = r.layer2(x)
        g = r.layer3(x)
        g = r.layer4(g)
        g = r.avgpool(g).flatten(1)
        return g, x

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']
        glob_src, feat_src = self.encode(src)
        glob_tgt, feat_tgt = self.encode(tgt)
        U_src = feature_align(feat_src, P_src, ns_src, self.rescale)
        U_tgt = feature_align(feat_tgt, P_tgt, ns_tgt, self.rescale)
        F_src = torch.cat([
            U_src,
            glob_src.expand(*glob_src.shape[:-1], U_src.shape[-1]),
            glob_tgt.expand(*glob_tgt.shape[:-1], U_src.shape[-1])
        ], 1)
        F_tgt = torch.cat([
            U_tgt,
            glob_tgt.expand(*glob_tgt.shape[:-1], U_tgt.shape[-1]),
            glob_src.expand(*glob_src.shape[:-1], U_tgt.shape[-1])
        ], 1)
        y_src = self.cls(F_src)
        y_tgt = self.cls(F_tgt)
        sim = torch.einsum("bci,bcj->bij", y_src, y_tgt)
        data_dict['ds_mat'] = self.sinkhorn(sim, ns_src, ns_tgt, dummy_row=True)
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        loss = 0.0
        if 'gt_perm_mat' in data_dict:
            if random.random() < 0.01:
                import numpy
                numpy.set_printoptions(formatter={"float": lambda x: "%.2f" % x if abs(x) > 0.01 else '----'})
                print(data_dict['ds_mat'][0, :ns_src[0], :ns_tgt[0]].detach().cpu().numpy())
                print(data_dict['gt_perm_mat'][0, :ns_src[0], :ns_tgt[0]].detach().cpu().numpy())
        data_dict['loss'] = loss
        # if 'gt_perm_mat' in data_dict:
        #     align = data_dict['gt_perm_mat'].argmax(-1)
        #     y_tgt_rand = y_tgt[..., torch.randperm(y_tgt.shape[-1]).to(align)]
        #     y_tgt_pm = y_tgt[torch.arange(len(y_tgt)).unsqueeze(1).to(align), ..., align].transpose(-1, -2)
        #     data_dict['loss'] = F.kl_div(y_src, y_tgt_pm) - F.kl_div(y_src, y_tgt_rand)
        # lab_src = y_src.argmax(1).cpu().numpy()
        # lab_tgt = y_tgt.argmax(1).cpu().numpy()
        # perm_mat = torch.zeros(len(lab_src), lab_src.shape[-1], lab_tgt.shape[-1])
        # for b in range(len(perm_mat)):
        #     for i in range(lab_src.shape[-1]):
        #         for j in range(lab_tgt.shape[-1]):
        #             if lab_src[b, i] == lab_tgt[b, j]:
        #                 perm_mat[b, i, j] = 1
        #                 lab_tgt[b, j] = -1
        #                 break
        # data_dict['perm_mat'] = perm_mat.to(y_src)
        # data_dict['ds_mat'] = perm_mat.to(y_tgt)
        return data_dict
