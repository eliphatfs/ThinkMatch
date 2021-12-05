import torch.optim as optim
import time
import xlwt
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from src.dataset.data_loader import GMDataset, get_dataloader
from src.displacement_layer import Displacement
from src.loss_func import *
from src.evaluation_metric import matching_accuracy
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval import eval_model
from src.lap_solvers.hungarian import hungarian
from src.utils.data_to_cuda import data_to_cuda
from extra.my_focal import MyFocalLoss

from src.utils.config import cfg, cfg_from_file
from pygmtools.benchmark import Benchmark
import numpy
from models.IGM.model import Net
import torch.nn as nn
import collections
import matplotlib.pyplot as plotlib
plotlib.style.use("seaborn")


def spot_dict(model, kv):
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            kv[name].append((layer.weight ** 2).mean().item())

def spot():
    cfg_from_file("experiments/igm.yaml")
    net = Net()
    kv = collections.defaultdict(list)
    for i in range(1, 10):
        load_model(net, "output/igm_voc/params/params_%04d.pt" % i)
        spot_dict(net, kv)
    plotlib.figure(figsize=[30, 30])
    for k, v in kv.items():
        plotlib.plot(numpy.array(v) / v[0], label=k)
    plotlib.legend()
    plotlib.savefig("spot.png")


if __name__ == "__main__":
    spot()
