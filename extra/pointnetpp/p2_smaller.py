import torch.nn as nn
import torch
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        additional_channel = 1024
        self.normal_channel = True
        self.sa1 = PointNetSetAbstractionMsg(8, [0.1, 0.2, 0.4], [8, 12, 16], 3 + additional_channel, [[128, 256], [256, 512], [64, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024 + 3, mlp=[1024, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=2048, mlp=[1024, 1024])
        self.fp1 = PointNetFeaturePropagation(in_channel=1030 + additional_channel, mlp=[512, 512])
        self.conv1 = nn.Conv1d(512, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 32, 1)

    def forward(self, xyz):
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
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss