from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import cfg
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.sinkhorn import Sinkhorn
from extra.pointnetpp import p2_smaller
from models.BBGM.sconv_archs import SiameseSConvOnNodes
from src.loss_func import PermutationLoss


loss_fn = PermutationLoss()
FF = F


class ResCls(nn.Module):
    # BN + ReLU Residually connected MLP
    def __init__(self, n, intro, unit, outro, ndim=1):
        super().__init__()
        BN = [nn.BatchNorm1d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][ndim]
        CN = [lambda x, y, _: nn.Linear(x, y), nn.Conv1d, nn.Conv2d, nn.Conv3d][ndim]
        self.verse = nn.ModuleList([BN(unit) for _ in range(n)])
        self.chorus = nn.ModuleList([CN(unit, unit, 1) for _ in range(n)])
        self.intro = CN(intro, unit, 1)
        self.outro = CN(unit, outro, 1)

    def forward(self, x):
        x = self.intro(x)
        for chorus, verse in zip(self.chorus, self.verse):
            d = torch.relu(verse(x))
            d = chorus(d)
            x = x + d
        return self.outro(x)


def my_align(raw_feature, P, ori_size: tuple):
    # Fast bilinear sampling by utilizing spatial transformer `grid_sample` operator
    return F.grid_sample(
        raw_feature,
        2 * P.unsqueeze(-2) / ori_size[0] - 1,
        'bilinear',
        'border',
        align_corners=False
    ).squeeze(-1)


def batch_features(embeddings, num_vertices):
    # Utility function for interfacing graph convolutions
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


def unbatch_features(orig, embeddings, num_vertices):
    # Utility function for interfacing graph convolutions
    res = torch.zeros_like(orig)
    cum = 0
    for embedding, num_v in zip(res, num_vertices):
        embedding[:, :num_v] = embeddings[cum: cum + num_v].t()
        cum = cum + num_v
    return res


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512)
        self.sconv = SiameseSConvOnNodes(256)
        self.pix2pt_proj = ResCls(1, feature_lat, 512, 256)
        self.pix2cl_proj = ResCls(1, 1024, 512, 128)
        self.edge_gate = ResCls(1, feature_lat * 3, 512, 1)
        self.edge_proj = ResCls(1, feature_lat * 3, 512, 64)
        self.tau = cfg.IGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pn = p2_smaller.get_model(256, 128, 64)
        self.masker = nn.Conv1d(256, 1, 1)
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.IGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.IGM.SK_EPSILON
        )
        self.backbone_params = list(self.resnet.parameters())

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        # an iterator through resnet features in a scale hierarchy
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

    def get_feats(self, feat_srcs, feat_tgts, P_src, P_tgt):
        # Bilinear sampling from grid features on keypoints
        # and concatenate different scales to get final point features
        U_src = torch.cat([
            my_align(feat_src, P_src, self.rescale) for feat_src in feat_srcs
        ], 1)
        U_tgt = torch.cat([
            my_align(feat_tgt, P_tgt, self.rescale) for feat_tgt in feat_tgts
        ], 1)
        glob_src = feat_srcs[-1].flatten(1).unsqueeze(-1)
        glob_tgt = feat_tgts[-1].flatten(1).unsqueeze(-1)
        # Global-level features are concatenated for cross-learning
        ghalo_src = torch.cat((glob_src, glob_tgt), 1)
        ghalo_tgt = torch.cat((glob_tgt, glob_src), 1)
        return U_src, U_tgt, ghalo_src, ghalo_tgt

    def edge_activations(self, feats, F, P, n):
        # F: BCN, P: BN2, n: B
        # Grab middle points between keypoints as edge features
        ep = ((P.unsqueeze(-2) + P.unsqueeze(-3)) / 2).flatten(1, 2)  # B N^2 2
        # ep are all middle points between keypoints pair-wise
        L = (
            torch.cat([F, torch.zeros_like(F)], 1).unsqueeze(-1) +
            torch.cat([torch.zeros_like(F), F], 1).unsqueeze(-2)
        ).flatten(2)  # B2CN^2
        # L is concatenated endpoint features from before
        E = torch.cat([
            my_align(feat, ep, self.rescale) for feat in feats
        ], 1)  # BCN^2
        # E is bilinear sampled CNN features on middle points
        CE = torch.cat([L, E], 1)
        # Concatenate the three parts
        mask = torch.arange(F.shape[-1], device=n.device).expand(len(F), F.shape[-1]) < n.unsqueeze(-1)
        # BN
        mask = mask.unsqueeze(-2) & mask.unsqueeze(-1)
        # Mask out non-existent keypoints (whose index >= n_points)
        # Projection with MLP from the three feature parts into final edge features
        # Also has normalization and gating for better stability
        return (torch.sigmoid(self.edge_gate(CE)) * FF.normalize(self.edge_proj(CE), dim=1) * mask.flatten(1).unsqueeze(1)).reshape(F.shape[0], -1, F.shape[-1], F.shape[-1])
    
    def points(self, y_src, y_tgt, P_src, P_tgt, n_src, n_tgt, e_src, e_tgt, g):
        resc = P_src.new_tensor(self.rescale)
        P_src, P_tgt = P_src / resc, P_tgt / resc
        P_src, P_tgt = P_src.transpose(1, 2), P_tgt.transpose(1, 2)
        key_mask_src = torch.arange(y_src.shape[-1], device=n_src.device).expand(len(y_src), y_src.shape[-1]) < n_src.unsqueeze(-1)
        key_mask_tgt = torch.arange(y_tgt.shape[-1], device=n_tgt.device).expand(len(y_tgt), y_tgt.shape[-1]) < n_tgt.unsqueeze(-1)
        key_mask_cat = torch.cat((key_mask_src, key_mask_tgt), -1).unsqueeze(1)
        # Mask for non-existent keypoints (whose index >= n_points)
        P_src = torch.cat((P_src, torch.zeros_like(P_src[:, :1])), 1)
        P_tgt = torch.cat((P_tgt, torch.ones_like(P_tgt[:, :1])), 1)
        # Merge features from parties
        pcd = torch.cat((P_src, P_tgt), -1)
        y_cat = torch.cat((y_src, y_tgt), -1)
        e_cat = torch.zeros([
            e_src.shape[0], e_src.shape[1],
            e_src.shape[2] + e_tgt.shape[2], e_src.shape[3] + e_tgt.shape[3]
        ], dtype=e_src.dtype, device=e_src.device)
        e_cat[..., :e_src.shape[2], :e_src.shape[3]] = e_src
        e_cat[..., e_src.shape[2]:, e_src.shape[3]:] = e_tgt
        # call point feature propagator with masked point and edge features
        r1, r2 = self.pn(torch.cat((pcd, y_cat), 1) * key_mask_cat, e_cat, g)
        # gather the features for current party
        return r1[..., :y_src.shape[-1]], r2[..., :y_src.shape[-1]]

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = [], []
        # Stacking two images into one batch for faster computation
        # Calls CNN encoder in function `encode`
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        # Jitter points by sub-pixel to avoid sampling artifacts
        if self.training:
            P_src = P_src + torch.rand_like(P_src) * 2 - 1
            P_tgt = P_tgt + torch.rand_like(P_tgt) * 2 - 1
        
        # Get (full) keypoints and global features
        F_src, F_tgt, g_src, g_tgt = self.get_feats(feat_srcs, feat_tgts, P_src, P_tgt)

        # Get edge features
        ea_src = self.edge_activations(feat_srcs, F_src, P_src, ns_src)
        ea_tgt = self.edge_activations(feat_tgts, F_tgt, P_tgt, ns_tgt)

        # MLP projection for downstream keypoint features
        y_src, y_tgt = self.pix2pt_proj(F_src), self.pix2pt_proj(F_tgt)

        # MLP projection for downstream global features
        g_src, g_tgt = self.pix2cl_proj(g_src), self.pix2cl_proj(g_tgt)

        # Normalize the features for better stability
        y_src, y_tgt = F.normalize(y_src, dim=1), F.normalize(y_tgt, dim=1)
        g_src, g_tgt = F.normalize(g_src, dim=1), F.normalize(g_tgt, dim=1)

        # Uncomment for SplineConv after getting the CNN features
        '''G_src, G_tgt = data_dict['pyg_graphs']
        G_src.x = batch_features(y_src, ns_src)
        G_src = self.sconv(G_src)
        G_tgt.x = batch_features(y_tgt, ns_tgt)
        G_tgt = self.sconv(G_tgt)
        y_src = unbatch_features(y_src, G_src.x, ns_src)
        y_tgt = unbatch_features(y_tgt, G_tgt.x, ns_tgt)'''
        
        # Call PointNet++ based feature propagation
        ff_src, folding_src = self.points(y_src, y_tgt, P_src, P_tgt, ns_src, ns_tgt, ea_src, ea_tgt, g_src)
        ff_tgt, folding_tgt = self.points(y_tgt, y_src, P_tgt, P_src, ns_tgt, ns_src, ea_tgt, ea_src, g_tgt)

        # Simple dot-product affinity
        sim = torch.einsum(
            'bci,bcj->bij',
            folding_src,
            folding_tgt
        )
        # Sinkhorn and output
        data_dict['ds_mat'] = self.sinkhorn(sim, ns_src, ns_tgt, dummy_row=True)
        mask_src = self.masker(ff_src)  # B x s
        mask_tgt = self.masker(ff_tgt)  # B x t
        # B x s x t
        data_dict['ds_mat'] = data_dict['ds_mat'] * mask_src.unsqueeze(-1) * mask_tgt.unsqueeze(-2)
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        # Output some of the useful features for possible integration with other modules
        # Not used when running stand-alone
        data_dict['ff'] = [ff_src, ff_tgt]
        data_dict['rf'] = [y_src, y_tgt]
        data_dict['gf'] = [g_src, g_tgt]
        return data_dict
