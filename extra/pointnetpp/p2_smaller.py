import torch.nn as nn
import torch
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from pygmtools.dataset import VOC2011_KPT_NAMES


labels = sorted(VOC2011_KPT_NAMES)


class MultiScalePropagation(nn.Module):
    def __init__(self, additional_channel, g_channel, o_channel):
        super().__init__()
        self.sa = PointNetSetAbstractionMsg(24, [0.1, 0.2, 0.3, 0.6, 1.0], [24] * 5, 3 + additional_channel, [[64, 128], [128, 256], [64, 128], [32, 64], [24, 48]])
        self.fp = PointNetFeaturePropagation(128 + 256 + 128 + 64 + 48 + 3 + additional_channel + g_channel, mlp=[512, 384])
        self.proj = nn.Conv1d(384, o_channel, 1)

    def forward(self, xyz, g):
        B, C, N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa(l0_xyz, l0_points)
        g = g.repeat(1, 1, N)
        l0_points = self.fp(l0_xyz, l1_xyz, torch.cat([g, l0_points], 1), l1_points)
        return self.proj(l0_points)


class HALOAttention(nn.Module):
    def __init__(self, sinkhorn, exff, emb_c, head_c):
        super().__init__()
        self.QK = nn.Conv1d(emb_c, head_c, 1)
        self.exff = exff
        self.sinkhorn = sinkhorn
        self.norm_src_1 = nn.InstanceNorm1d(emb_c)
        self.norm_tgt_1 = nn.InstanceNorm1d(emb_c)
        self.norm_src_2 = nn.InstanceNorm1d(emb_c)
        self.norm_tgt_2 = nn.InstanceNorm1d(emb_c)

    def forward(self, x_src, x_tgt, n_src, n_tgt):
        # BCS, BCT
        qk_src = self.QK(x_src)
        qk_tgt = self.QK(x_tgt)
        act = torch.einsum("bcs,bct->bst", qk_src, qk_tgt)
        attention = self.sinkhorn(act, n_src, n_tgt, dummy_row=True)
        copied_src = torch.einsum("bst,bct->bcs", attention, x_tgt)
        copied_tgt = torch.einsum("bst,bcs->bct", attention, x_src)
        x_src = self.norm_src_1(x_src + copied_src)
        x_tgt = self.norm_tgt_1(x_tgt + copied_tgt)
        x_src = self.norm_src_2(self.exff(x_src))
        x_tgt = self.norm_tgt_2(self.exff(x_tgt))
        return x_src, x_tgt


class get_model(nn.Module):
    def __init__(self, additional_channel, g_channel):
        super(get_model, self).__init__()
        self.normal_channel = True
        self.sa1 = PointNetSetAbstractionMsg(36, [0.1, 0.2, 0.3, 0.6, 1.0], [36] * 5, 3 + additional_channel, [[64, 128], [128, 256], [64, 128], [32, 64], [24, 48]])
        self.sa2 = PointNetSetAbstractionMsg(36, [0.1, 0.2, 0.3, 0.6, 1.0], [36] * 5, 512 + 64 + 48, [[96, 128], [192, 256], [96, 128], [48, 64], [36, 48]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 64 + 48 + 3, mlp=[300, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 128 + 256 + 128 + 64 + 48, mlp=[1024, 512])
        self.fp2 = PointNetFeaturePropagation(in_channel=512 + 128 + 256 + 128 + 64 + 48, mlp=[512, 384])
        self.fp1 = PointNetFeaturePropagation(in_channel=384 + 6 + 32 * 0 + g_channel + additional_channel, mlp=[512, 256])
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(g_channel, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Conv1d(512, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 5, 1)
            )
            for _ in range(4)
        ])
        self.conv1 = nn.Conv1d(256, 32, 1)
        self.cls_emb = nn.Embedding(len(labels), 32)

    def forward(self, xyz, g):
        # Set Abstraction layers
        attns = [*map(lambda f: F.softmax(f(g).flatten(1), dim=1), self.scale_attentions)]
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, attns[0] + attns[1])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, attns[2] + attns[3])
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # cls_label = self.cls_emb(torch.tensor([labels.index(i) for i in cls], device=l1_points.device))
        # cls_label_one_hot = cls_label.view(B, 32, 1).repeat(1, 1, N)
        g = g.repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([g, l0_xyz, l0_points], 1), l1_points)
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
