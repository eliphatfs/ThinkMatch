import torch

def my_align(*args): ...

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
