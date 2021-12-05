import torch.optim as optim
import time
from src.dataset import data_loader
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


def spot_hook(model, kv):
    def _hook(clos_name):
        def __hook(module, i, o):
            kv[clos_name].extend([o.mean().item(), o.std().item(), o.min().item(), o.max().item()])
    
        return __hook

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            layer.register_forward_hook(_hook(name))


def spot_norm():
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


def spot_out_minmax():
    cfg_from_file("experiments/igm.yaml")
    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    ds_dict = cfg[cfg.DATASET_FULL_NAME] if ('DATASET_FULL_NAME' in cfg) and (cfg.DATASET_FULL_NAME in cfg) else {}
    benchmark = {
        x: Benchmark(name=cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     obj_resize=cfg.PROBLEM.RESCALE,
                     filter=cfg.PROBLEM.FILTER,
                     **ds_dict)
        for x in ('train', 'test')}
    image_dataset = {
        x: GMDataset(name=cfg.DATASET_FULL_NAME,
                     bm=benchmark[x],
                     problem=cfg.PROBLEM.TYPE,
                     length=dataset_len[x],
                     cls=cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     using_all_graphs=cfg.PROBLEM.TRAIN_ALL_GRAPHS if x == 'train' else cfg.PROBLEM.TEST_ALL_GRAPHS)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], shuffle=True, fix_seed=(x == 'test'), batch_size=cfg.BATCH_SIZE if (x != 'test') else 32)
                  for x in ('train', 'test')}
    net = Net()
    kv = collections.defaultdict(list)
    load_model(net, "output/igm_voc/params/params_%04d.pt" % 10)
    net.cuda()
    net.eval()
    spot_hook(net, kv)
    plotlib.figure(figsize=[54, 24])

    def _do(split):
        for i, inputs in enumerate(dataloader[split]):
            if i >= 10:
                break
            inputs = data_to_cuda(inputs)
            with torch.no_grad():
                net(inputs)
        kvs = []
        labels = []
        for k, v in kv.items():
            kvs.append(numpy.array(v))
            labels.extend([k])
        plotlib.boxplot(numpy.array(kvs).T, labels=labels)
        plotlib.xticks(rotation = -90)
        plotlib.xlabel(split)

    plotlib.subplot(2, 1, 1)
    _do("train")
    plotlib.subplot(2, 1, 2)
    _do("test")
    plotlib.savefig("spot.png")

if __name__ == "__main__":
    spot_out_minmax()
