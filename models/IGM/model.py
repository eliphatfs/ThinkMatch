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
        self.cls = ResCls(4, feature_lat, 2048, 512)
        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pos_emb = torch.nn.Parameter(torch.randn(512, 16, 16))
        self.attentions = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(512, 8)
            for _ in range(4)
        ])
        self.atn_mlp = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.LayerNorm(512), torch.nn.ReLU())
            for _ in range(4)
        ])
        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)

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

    def attn(self, y_src, y_tgt, P_src, P_tgt, n_src, n_tgt):
        exp_posemb = self.pos_emb.expand(len(y_src), *self.pos_emb.shape)
        y_src = (y_src + my_align(exp_posemb, P_src, self.rescale)).permute(2, 0, 1)
        y_tgt = (y_tgt + my_align(exp_posemb, P_tgt, self.rescale)).permute(2, 0, 1)
        import pdb; pdb.set_trace()
        key_mask_src = torch.arange(y_src.shape[-1]).expand(len(y_src), y_src.shape[-1]) < n_src.unsqueeze(-1)
        key_mask_tgt = torch.arange(y_tgt.shape[-1]).expand(len(y_tgt), y_tgt.shape[-1]) < n_tgt.unsqueeze(-1)
        for atn, ff in zip(self.attentions, self.atn_mlp):
            atn_src, _ = atn(y_src, y_tgt, y_tgt, key_padding_mask=key_mask_tgt)
            atn_tgt, _ = atn(y_tgt, y_src, y_src, key_padding_mask=key_mask_src)
            y_src = y_src + ff(atn_src)
            y_tgt = y_tgt + ff(atn_tgt)
        return y_src.permute(1, 2, 0), y_tgt.permute(1, 2, 0)

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = list(self.encode(src)), list(self.encode(tgt))
        F_src, F_tgt = self.halo(feat_srcs, feat_tgts, P_src, P_tgt)

        y_src, y_tgt = self.cls(F_src), self.cls(F_tgt)
        y_src, y_tgt = self.attn(y_src, y_tgt, P_src, P_tgt, ns_src, ns_tgt)

        sim = torch.einsum(
            "bci,bcj->bij",
            y_src - y_src.mean(-1, keepdim=True),
            y_tgt - y_tgt.mean(-1, keepdim=True)
        )
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
