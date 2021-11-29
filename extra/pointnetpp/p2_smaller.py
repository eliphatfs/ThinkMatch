import torch.nn as nn
import torch
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from pygmtools.dataset import VOC2011_KPT_NAMES


labels = sorted(VOC2011_KPT_NAMES)


class get_model(nn.Module):
    def __init__(self, additional_channel, g_channel):
        super(get_model, self).__init__()
        self.normal_channel = True
        self.sa1 = PointNetSetAbstractionMsg(24, [0.1, 0.2, 0.3], [24, 24, 24], 3 + additional_channel, [[64, 128], [128, 256], [64, 128]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 128 + 256 + 128, mlp=[1024, 512])
        self.fp1 = PointNetFeaturePropagation(in_channel=518 + 32 + g_channel + additional_channel, mlp=[512, 256])
        self.conv1 = nn.Conv1d(256, 32, 1)
        self.cls_emb = nn.Embedding(len(labels), 32)

    def forward(self, xyz, cls, g):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l3_xyz, l3_points = self.sa3(l1_xyz, l1_points)
        # Feature Propagation layers
        l1_points = self.fp3(l1_xyz, l3_xyz, l1_points, l3_points)
        cls_label = self.cls_emb(torch.tensor([labels.index(i) for i in cls], device=l1_points.device))
        cls_label_one_hot = cls_label.view(B, 32, 1).repeat(1, 1, N)
        g = g.repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, g, l0_xyz, l0_points], 1), l1_points)
        # FC layers
        x = self.conv1(l0_points)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss