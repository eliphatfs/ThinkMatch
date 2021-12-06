import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class KNN(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def forward(self, xyz1, xyz2):
        dist = square_distance(xyz1, xyz2).sqrt()
        return torch.topk(dist, self.k, dim=-1, largest=False, sorted=False)


class Interpolate(nn.Module):
    def __init__(self):
        super().__init__()

    # features: B * M * C, idx: B * N * K
    def forward(self, features, idx, weight):
        B, N, K = idx.size()
        features_gathered = torch.gather(features, 1, idx.view(B, -1, 1).expand(-1, -1, features.size(-1))).view(B, N, K, -1)  # B * N * K * C
        return (features_gathered * weight[..., None]).sum(2)  # B * N * C



def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, nsample, xyz, points, density_scale=None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density


def sample_and_group_all(xyz, points, density_scale=None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # new_xyz = torch.zeros(B, 1, C).to(device)
    new_xyz = xyz.mean(dim=1, keepdim=True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = density_scale.view(B, 1, N, 1)
        return new_xyz, new_points, grouped_xyz, grouped_density


def group(nsample, xyz, points, density_scale=None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_points, grouped_xyz_norm
    else:
        grouped_density = index_points(density_scale, idx)
        return new_points, grouped_xyz_norm, grouped_density


def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    # import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim=-1)

    return xyz_density


class DensityNet(nn.Module):
    def __init__(self, hidden_unit=[8, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv1d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv1d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm1d(1))

    def forward(self, xyz_density):
        B, N = xyz_density.shape
        density_scale = xyz_density.unsqueeze(1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale = bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale) + 0.5
            else:
                density_scale = F.relu(density_scale)

        return density_scale


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))

        return weights


class PointConvDensitySetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet()
        self.group_all = group_all
        self.bandwidth = bandwidth

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = compute_density(xyz, self.bandwidth)
        # import ipdb; ipdb.set_trace()
        density_scale = self.densitynet(xyz_density)

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(xyz, points,
                                                                                          density_scale.view(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(min(self.npoint, N), min(self.nsample, N), xyz,
                                                                                         points,
                                                                                         density_scale.view(B, N, 1))
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        grouped_xyz = grouped_xyz * grouped_density.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                min(self.npoint, N),
                                                                                                                -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointConvDensityFeaturePropogation(nn.Module):
    def __init__(self, nsample, in_channel, concat_channel, mlp, bandwidth):
        super(PointConvDensityFeaturePropogation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.nsample = nsample

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * in_channel, mlp[0])
        self.bn_linear = nn.BatchNorm1d(mlp[0])

        self.mlps = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = mlp[0] + concat_channel
        for out_channel in mlp[1:]:
            self.mlps.append(nn.Linear(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.densitynet = DensityNet()
        self.bandwidth = bandwidth
        self.knn = KNN()
        self.interp = Interpolate()

    def forward(self, xyz1, xyz2, points1, points2):
        B = xyz1.shape[0]
        N = xyz1.shape[2]
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        dist, idx = self.knn(xyz1, xyz2)
        dist = torch.max(dist, torch.tensor(1e-10).cuda())
        norm = torch.sum(1. / dist, dim=2, keepdim=True)
        weight = (1. / dist) / norm
        interpolated_points = self.interp(points2, idx, weight)

        xyz1_density = compute_density(xyz1, self.bandwidth)
        # import ipdb; ipdb.set_trace()
        density_scale = self.densitynet(xyz1_density)

        new_points, grouped_xyz_norm, grouped_density = group(min(self.nsample, N), xyz1, interpolated_points, density_scale.view(B, N, 1))
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        grouped_xyz = grouped_xyz * grouped_density.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                N,
                                                                                                                -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points).permute(0, 2, 1)
        if points1 is not None:
            new_points = torch.cat([new_points, points1], dim=2)

        for i, mlp in enumerate(self.mlps):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(mlp(new_points).permute(0, 2, 1))).permute(0, 2, 1)

        # new_xyz = xyz1.permute(0, 2, 1)

        return new_points.permute(0, 2, 1)


class PointConv(nn.Module):
    def __init__(self, g_channel, i_channel):
        super().__init__()
        self.sa1 = PointConvDensitySetAbstraction(npoint=32, nsample=32, in_channel=i_channel + 3, mlp=[32, 32, 64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=32, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], bandwidth = 0.3, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=32, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.6, group_all=False)
        # self.sa4 = PointConvDensitySetAbstraction(npoint=36, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], bandwidth = 0.8, group_all=False)
        # self.fp1 = PointConvDensityFeaturePropogation(nsample=16, in_channel=512+3,  concat_channel=256, bandwidth=0.8, mlp=[512, 512])
        self.fp2 = PointConvDensityFeaturePropogation(nsample=16, in_channel=256+3, concat_channel=128, bandwidth=0.3, mlp=[256, 256])
        self.fp3 = PointConvDensityFeaturePropogation(nsample=16, in_channel=256+3, concat_channel=64, bandwidth=0.2, mlp=[256, 128])
        self.fp4 = PointConvDensityFeaturePropogation(nsample=16, in_channel=128+3, concat_channel=3+g_channel+i_channel, bandwidth=0.1, mlp=[128, 64, 64])

        self.mlp = nn.Conv1d(64, 32, 1)

    def forward(self, xyz, g):
        _, _, N = xyz.shape
        l0_xyz, l0_points = xyz[:, :3], xyz[:, 3:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # l3_points = self.fp1(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp3(l1_xyz, l2_xyz, l1_points, l2_points)
        g = g.repeat(1, 1, N)
        l0_points = self.fp4(l0_xyz, l1_xyz, torch.cat([g, l0_points], 1), l1_points)

        return self.mlp(l0_points)
