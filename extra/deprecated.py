import torch
import sys

def my_align(*args): ...
def my_cdist2(*args): ...

class Module:
    def __init__(self) -> None:
        super().__init__()
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512 * 2)
        self.pos_emb = torch.nn.Parameter(torch.randn(feature_lat, 16, 16))
        self.attentions = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(512, 8)
            for _ in range(4)
        ])
        self.atn_mlp = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(512, 1536), torch.nn.GELU(), torch.nn.Linear(1536, 512))
            for _ in range(4)
        ])
        self.atn_norm = torch.nn.LayerNorm(512)

    def attn(self, y_src, y_tgt, P_src, P_tgt, n_src, n_tgt):
        exp_posemb = self.pos_emb.expand(len(y_src), *self.pos_emb.shape)
        key_mask_src = torch.arange(y_src.shape[-1], device=n_src.device).expand(len(y_src), y_src.shape[-1]) >= n_src.unsqueeze(-1)
        key_mask_tgt = torch.arange(y_tgt.shape[-1], device=n_tgt.device).expand(len(y_tgt), y_tgt.shape[-1]) >= n_tgt.unsqueeze(-1)
        key_mask_cat = torch.cat((key_mask_src, key_mask_tgt), -1)
        atn_mask_cat = (key_mask_cat.unsqueeze(-1) & key_mask_cat.unsqueeze(-2)).unsqueeze(1).repeat_interleave(8, 1).flatten(0, 1)
        y_src = (y_src + my_align(exp_posemb, P_src, self.rescale)).permute(2, 0, 1)
        y_tgt = (y_tgt + my_align(exp_posemb, P_tgt, self.rescale)).permute(2, 0, 1)
        y_cat = torch.cat((y_src, y_tgt), 0)
        for atn, ff in zip(self.attentions, self.atn_mlp):
            x = self.atn_norm(y_cat)
            atn_cat, atn_wei = atn(x, x, x, key_padding_mask=key_mask_cat, attn_mask=atn_mask_cat)
            y_cat = y_cat + atn_cat
            x = self.atn_norm(y_cat)
            y_cat = y_cat + ff(x)
        return atn_wei[..., :len(y_src), len(y_src):] + atn_wei[..., len(y_src):, :len(y_src)].transpose(1, 2)

        # sim = self.attn(y_src, y_tgt, P_src, P_tgt, ns_src, ns_tgt)
        # sim = self.projection(y_src), self.projection(y_tgt)
        # sim = torch.einsum(
        #     "bci,bcj->bij",
        #     e_src, e_tgt
        #     # y_src - y_src.mean(-1, keepdim=True),
        #     # y_tgt - y_tgt.mean(-1, keepdim=True)
        # )

    def fold(self, data_dict, y_src, y_tgt, P_src, P_tgt, ns_src, ns_tgt):
        folding_src = self.points(y_src, y_tgt, P_src, P_tgt, 64 + ns_src, 64 + ns_tgt)
        folding_tgt = self.points(y_tgt, y_src, P_tgt, P_src, 64 + ns_tgt, 64 + ns_src)
        # for b in range(len(y_src)):
        #     folding_src[b, ns_src[b]:] = folding_src[b, :ns_src[b]].mean(-2, keepdim=True)
        #     folding_tgt[b, ns_tgt[b]:] = folding_tgt[b, :ns_tgt[b]].mean(-2, keepdim=True)
        # folding_src = (folding_src - folding_src.min(-2, keepdim=True)[0]) / (folding_src.max(-2, keepdim=True)[0] - folding_src.min(-2, keepdim=True)[0] + 1e-8)
        # folding_tgt = (folding_tgt - folding_tgt.min(-2, keepdim=True)[0]) / (folding_tgt.max(-2, keepdim=True)[0] - folding_tgt.min(-2, keepdim=True)[0] + 1e-8)
        for b in range(len(y_src)):
            folding_src[b, ns_src[b]:] = torch.randn_like(folding_src[b, ns_src[b]:])
            folding_tgt[b, ns_tgt[b]:] = torch.randn_like(folding_src[b, ns_tgt[b]:])
        # sim = torch.zeros(y_src.shape[0], y_src.shape[-1] - 64, y_tgt.shape[-1] - 64).to(y_src)
        # for b in range(len(y_src)):
        #     sim[b, :ns_src[b], :ns_tgt[b]] = torch.clamp(self.ot(
        #         folding_src[b: b + 1, :ns_src[b]],
        #         folding_tgt[b: b + 1, :ns_tgt[b]],
        #     )[1].squeeze(0) * torch.min(ns_src[b], ns_tgt[b]), 0, 1)
        sim = my_cdist2(folding_src, folding_tgt)
        ds_src = my_cdist2(folding_src, folding_src)
        ds_tgt = my_cdist2(folding_tgt, folding_tgt)
        bi = torch.arange(len(folding_src), device=folding_tgt.device).unsqueeze(-1)
        dist = (((folding_src - folding_tgt[bi, data_dict['gt_perm_mat'][0].argmax(-1)]) ** 2).sum(-1) + 1e-8) ** 0.5
        data_dict['loss'] = (
            - (1 / (1e-7 + dist)).mean()
            + (1 / (1e-7 + ds_src.topk(2, dim=1, largest=False)[0][:, -1])).mean()
            + (1 / (1e-7 + ds_tgt.topk(2, dim=1, largest=False)[0][:, -1])).mean()
        )
        if torch.rand(1) < 0.005:
            print("S = ", file=sys.stderr)
            print((sim[0] / sim[0].max()).detach().cpu().numpy(), file=sys.stderr)
            print("G = ", file=sys.stderr)
            print(data_dict['gt_perm_mat'][0].detach().cpu().numpy(), file=sys.stderr)
            print("Ps = ", file=sys.stderr)
            print(folding_src[0].t().detach().cpu().numpy(), file=sys.stderr)
            print("Pt = ", file=sys.stderr)
            print(folding_tgt[0][data_dict['gt_perm_mat'][0].argmax(-1)].t().detach().cpu().numpy(), file=sys.stderr)
        
