# file: embedding_averager.py
# from https://chatgpt.com/share/6845ec6c-b118-800b-88b3-3a2b0bf9351a
import torch
from torch import nn
from typing import Tuple

class EmbeddingAverager(nn.Module):
    """
    Compute the per-identity mean (or weighted mean) of an embedding batch.

    Parameters
    ----------
    reduction : str
        'mean' (default) or 'weighted'.
        When 'weighted', `weights` must be supplied in forward().
    return_index_map : bool
        If True, also returns a tensor that maps each original sample
        to its aggregated row – handy for gathering losses.
    """

    def __init__(self, reduction: str = "mean", return_index_map: bool = False):
        super().__init__()
        assert reduction in {"mean", "weighted"}
        self.reduction = reduction
        self.return_index_map = return_index_map

    @torch.no_grad()
    def forward(
        self,
        emb: torch.Tensor,             # shape (B, D)
        labels: torch.Tensor,          # shape (B,)
        weights: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        device = emb.device
        labels = labels.to(device)

        # ---- gather bookkeeping -------------------------------------------------
        # unique ids and position of each sample’s aggregated row
        uniq_ids, inv = torch.unique(labels, return_inverse=True)
        num_ids = uniq_ids.size(0)

        if self.reduction == "mean":
            # quick path – use scatter_add then divide by counts
            summed = torch.zeros(num_ids, emb.size(1), device=device).scatter_add_(0,
                      inv.unsqueeze(-1).expand_as(emb), emb)
            counts = torch.bincount(inv, minlength=num_ids).unsqueeze(1)
            agg = summed / counts.clamp_min(1)

        else:  # weighted mean
            assert weights is not None, "`weights` required for weighted reduction"
            weights = weights.to(device).unsqueeze(1)  # (B, 1)
            w_sum = torch.zeros(num_ids, 1, device=device).scatter_add_(0, inv.unsqueeze(-1), weights)
            w_emb = torch.zeros_like(summed).scatter_add_(0, inv.unsqueeze(-1).expand_as(emb), emb * weights)
            agg = w_emb / w_sum.clamp_min(1e-8)

        if self.return_index_map:
            return agg, uniq_ids, inv
        return agg, uniq_ids, None

