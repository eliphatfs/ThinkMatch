import torch
import torch.nn as nn
from torch import Tensor


class MyFocalLoss(nn.Module):
    r"""
    Focal loss between two permutations.

    .. math::
        L_{focal} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left((1-\mathbf{S}_{i,j})^{\gamma} \mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} +
        \mathbf{S}_{i,j}^{\gamma} (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs, :math:`\gamma` is the focal loss
    hyper parameter.

    :param gamma: :math:`\gamma` parameter for focal loss
    :param eps: a small parameter for numerical stability

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self, gamma=0., eps=1e-15):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged focal loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            x = pred_dsmat[b, :src_ns[b], :tgt_ns[b]]
            y = gt_perm[b, :src_ns[b], :tgt_ns[b]]
            loss += torch.sum(
                - (1 - x ** self.gamma) * y * torch.log(x + self.eps)
                - x ** self.gamma * (1 - y) * torch.log(1 - x + self.eps)
            )
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum