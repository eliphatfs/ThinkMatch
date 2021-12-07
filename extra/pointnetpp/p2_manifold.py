import torch.nn as nn
import torch
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, additional_channel, g_channel):
        super(get_model, self).__init__()
        self.normal_channel = True
        self.sa1 = PointNetSetAbstractionMsg(36, [0.1, 0.2, 0.4], [36] * 3, 3 + additional_channel, [[64, 128], [128, 256], [64, 128]])
        self.sa2 = PointNetSetAbstractionMsg(36, [0.1, 0.2, 0.3], [36] * 3, 512, [[64, 128], [128, 256], [64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(36, [0.1, 0.2, 0.3], [36] * 3, 512, [[64, 128], [128, 256], [64, 128]])
        # self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 64 + 48 + 3, mlp=[300, 1024], group_all=True)
        self.fp = PointNetFeaturePropagation(in_channel=512 + g_channel, mlp=[512, 256])
        self.conv = nn.Conv1d(256, 32, 1)

    def forward(self, xyz, g, tp):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        g = g.repeat(1, 1, tp.shape[-1])
        l0_points = self.fp(tp, l1_xyz, g, l1_points)
        # FC layers
        x = self.conv(l0_points)
        return x
