import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from models.BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from src.utils.pad_tensor import pad_tensor
from models.NGM.gnn import GNNLayer
from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.hungarian import hungarian
from extra.pointnetpp import p2_smaller

from src.utils.config import cfg
from torchvision.models import resnet34


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class ResCls(nn.Module):
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
    return F.grid_sample(
        raw_feature,
        2 * P.unsqueeze(-2) / ori_size[0] - 1,
        'bilinear',
        'border',
        align_corners=False
    ).squeeze(-1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.resnet = resnet34(True)  # UNet(3, 2)
        # self.unet.load_state_dict(torch.load("unet_carvana_scale0.5_epoch1.pth"))
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512 * 2)
        # self.sconv = SiameseSConvOnNodes(48)
        self.pix2pt_proj = ResCls(1, feature_lat, 512, 96)
        self.pix2cl_proj = ResCls(1, 1024, 512, 96)
        # self.edge_proj = ResCls(2, feature_lat * 3 - 512, 1024, 1)
        self.tau = cfg.IGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pn = p2_smaller.get_model(96, 96, 256)
        
        self.backbone_params = list(self.resnet.parameters())

        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.NGM.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.NGM.FEATURE_CHANNEL * 2
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.NGM.SK_TAU
        self.mgm_tau = cfg.NGM.MGM_SK_TAU
        self.univ_size = cfg.NGM.UNIV_SIZE

        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.sinkhorn_mgm = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, epsilon=cfg.NGM.SK_EPSILON, tau=self.mgm_tau)
        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            tau = cfg.NGM.SK_TAU
            if i == 0:
                gnn_layer = GNNLayer(1, 1,
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
            else:
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
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

    def halo(self, feat_srcs, feat_tgts, P_src, P_tgt):
        U_src = torch.cat([
            my_align(feat_src, P_src, self.rescale) for feat_src in feat_srcs
        ], 1)
        U_tgt = torch.cat([
            my_align(feat_tgt, P_tgt, self.rescale) for feat_tgt in feat_tgts
        ], 1)
        glob_src = feat_srcs[-1].flatten(1).unsqueeze(-1)
        glob_tgt = feat_tgts[-1].flatten(1).unsqueeze(-1)
        F_src = torch.cat([
            U_src,
            glob_tgt.expand(*glob_tgt.shape[:-1], U_src.shape[-1])
        ], 1)
        F_tgt = torch.cat([
            U_tgt,
            glob_src.expand(*glob_src.shape[:-1], U_tgt.shape[-1])
        ], 1)
        ghalo_src = torch.cat((glob_src, glob_tgt), 1)
        ghalo_tgt = torch.cat((glob_tgt, glob_src), 1)
        return F_src, F_tgt, ghalo_src, ghalo_tgt

    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['batch_size']
        num_graphs = len(images)

        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']
        feat_srcs, feat_tgts = [], []
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])

        feat_srcs, feat_tgts = [], []
        for feat in self.encode(torch.cat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        if self.training:
            P_src = P_src + torch.rand_like(P_src) * 2 - 1
            P_tgt = P_tgt + torch.rand_like(P_tgt) * 2 - 1
        F_src, F_tgt, g_src, g_tgt = self.halo(feat_srcs, feat_tgts, P_src, P_tgt)

        # ea_src = self.edge_activations(feat_srcs, F_src, P_src, ns_src)
        # ea_tgt = self.edge_activations(feat_tgts, F_tgt, P_tgt, ns_tgt)

        y_src, y_tgt = self.pix2pt_proj(F_src), self.pix2pt_proj(F_tgt)

        g_src, g_tgt = self.pix2cl_proj(g_src), self.pix2cl_proj(g_tgt)
        y_src, y_tgt = F.normalize(y_src, dim=1), F.normalize(y_tgt, dim=1)
        g_src, g_tgt = F.normalize(g_src, dim=1), F.normalize(g_tgt, dim=1)
        
        folding_src = self.points(y_src, y_tgt, P_src, P_tgt, ns_src, ns_tgt, g_src)
        folding_tgt = self.points(y_tgt, y_src, P_tgt, P_src, ns_tgt, ns_src, g_tgt)

        global_list = [g_src, g_tgt]
        orig_graph_list = []
        for feat, graph in zip([folding_src, folding_tgt], graphs):
            graph.x = feat
            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []

        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
            Kp = torch.stack(pad_tensor(unary_affs), dim=0)
            Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
            K = construct_aff_mat(Ke, Kp, kro_G, kro_H)
            if num_graphs == 2: data_dict['aff_mat'] = K

            if cfg.NGM.FIRST_ORDER:
                emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
            else:
                emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

            if cfg.NGM.POSITIVE_EDGES:
                A = (K > 0).to(K.dtype)
            else:
                A = (K != 0).to(K.dtype)

            emb_K = K.unsqueeze(-1)

            # NGM qap solver
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb_K, emb = gnn_layer(A, emb_K, emb, n_points[idx1], n_points[idx2])

            v = self.classifier(emb)
            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)

            ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
            x = hungarian(ss, n_points[idx1], n_points[idx2])
            s_list.append(ss)
            x_list.append(x)
            indices.append((idx1, idx2))

        if num_graphs > 2:
            joint_indices = torch.cat((torch.cumsum(torch.stack([torch.max(np) for np in n_points]), dim=0), torch.zeros((1,), dtype=torch.long, device=K.device)))
            joint_S = torch.zeros(batch_size, torch.max(joint_indices), torch.max(joint_indices), device=K.device)
            for idx in range(num_graphs):
                for b in range(batch_size):
                    start = joint_indices[idx-1]
                    joint_S[b, start:start+n_points[idx][b], start:start+n_points[idx][b]] += torch.eye(n_points[idx][b], device=K.device)

            for (idx1, idx2), s in zip(indices, s_list):
                if idx1 > idx2:
                    joint_S[:, joint_indices[idx2-1]:joint_indices[idx2], joint_indices[idx1-1]:joint_indices[idx1]] += s.transpose(1, 2)
                else:
                    joint_S[:, joint_indices[idx1-1]:joint_indices[idx1], joint_indices[idx2-1]:joint_indices[idx2]] += s

            matching_s = []
            for b in range(batch_size):
                e, v = torch.symeig(joint_S[b], eigenvectors=True)
                diff = e[-self.univ_size:-1] - e[-self.univ_size+1:]
                if self.training and torch.min(torch.abs(diff)) <= 1e-4:
                    matching_s.append(joint_S[b])
                else:
                    matching_s.append(num_graphs * torch.mm(v[:, -self.univ_size:], v[:, -self.univ_size:].transpose(0, 1)))

            matching_s = torch.stack(matching_s, dim=0)

            for idx1, idx2 in indices:
                s = matching_s[:, joint_indices[idx1-1]:joint_indices[idx1], joint_indices[idx2-1]:joint_indices[idx2]]
                s = self.sinkhorn_mgm(torch.log(torch.relu(s)), n_points[idx1], n_points[idx2]) # only perform row/col norm, do not perform exp
                x = hungarian(s, n_points[idx1], n_points[idx2])

                mgm_s_list.append(s)
                mgm_x_list.append(x)

        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0]
            })
        elif cfg.PROBLEM.TYPE == 'MGM':
            data_dict.update({
                'ds_mat_list': mgm_s_list,
                'perm_mat_list': mgm_x_list,
                'graph_indices': indices,
            })

        return data_dict
