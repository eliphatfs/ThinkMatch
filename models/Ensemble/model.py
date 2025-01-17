from models.IGM.model import Net as IGM
from models.NGM.model_v2 import Net as NGM
from models.BBGM.model import Net as BBGM
import torch
from src.lap_solvers.hungarian import hungarian


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.igm = IGM()
        self.igm.load_state_dict(torch.load("output/igm_voc/params/params_9999.pt"))
        self.ngm = NGM()
        self.ngm.load_state_dict(torch.load("pretrained_params_vgg16_ngmv2_voc.pt"))
        self.bbgm = BBGM()
        self.bbgm.load_state_dict(torch.load("output/vgg16_bbgm_voc/params/params_0019.pt"))

    @property
    def device(self):
        return self.igm.device

    def forward(self, x):
        x = self.igm(x)
        ds = x['ds_mat']
        x = self.ngm(x)
        ns_src, ns_tgt = x['ns']
        ds = ds + x['ds_mat']
        x = self.bbgm(x)
        ds = ds + x['perm_mat'] * 0.5
        x['perm_mat'] = hungarian(ds, ns_src, ns_tgt)
        return x
