from models.IGM.unet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.feature_align import feature_align
from src.utils.config import cfg


class ResCls(nn.Module):
    def __init__(self, n, unit, outro):
        super().__init__()
        self.verse = nn.ModuleList([nn.BatchNorm1d(unit) for _ in range(n)])
        self.chorus = nn.ModuleList([nn.Conv1d(unit, unit, 1) for _ in range(n)])
        self.outro = nn.Conv1d(unit, outro, 1)

    def forward(self, x):
        for chorus, verse in zip(self.chorus, self.verse):
            d = torch.relu(verse(x))
            d = chorus(d)
            x = x + d
        return self.outro(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(3, 128)
        self.rescale = cfg.PROBLEM.RESCALE
        self.cls = ResCls(3, 512, 24)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']
        feat_src = self.unet(src)
        feat_tgt = self.unet(tgt)
        glob_src = torch.max(feat_src.flatten(2), -1, True)[0]
        glob_tgt = torch.max(feat_tgt.flatten(2), -1, True)[0]
        U_src = feature_align(feat_src, P_src, ns_src, self.rescale)
        U_tgt = feature_align(feat_tgt, P_tgt, ns_tgt, self.rescale)
        U_src = torch.cat([U_src, glob_src.expand_as(U_src)], 1)
        U_tgt = torch.cat([U_tgt, glob_tgt.expand_as(U_tgt)], 1)
        F_src = torch.cat([U_src, U_tgt], 1)
        F_tgt = torch.cat([U_tgt, U_src], 1)
        y_src = F.softmax(self.cls(F_src), 1)
        y_tgt = F.softmax(self.cls(F_tgt), 1)
        if 'gt_perm_mat' in data_dict:
            align = data_dict['gt_perm_mat'].argmax(-1)
            y_tgt_rand = y_tgt[..., torch.randperm(y_tgt.shape[-1]).to(align)]
            y_tgt_pm = y_tgt[torch.arange(len(y_tgt)).unsqueeze(1).to(align), ..., align].transpose(-1, -2)
            data_dict['loss'] = F.kl_div(y_src, y_tgt_pm) - F.kl_div(y_src, y_tgt_rand)
        lab_src = y_src.argmax(1)
        lab_tgt = y_tgt.argmax(1)
        perm_mat = (lab_tgt.unsqueeze(-2) == lab_src.unsqueeze(-1)).to(y_src)
        data_dict['perm_mat'] = perm_mat
        data_dict['ds_mat'] = perm_mat
        import pdb; pdb.set_trace()
        return data_dict
