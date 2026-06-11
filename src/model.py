"""Batched Elhage toy models: many independent models as one tensor computation."""

import torch
from torch import nn


class BatchedToyModels(nn.Module):
    """M independent toy models x_hat = ReLU(W^T W x + b), trained in parallel.

    Model m has hidden width widths[m]. All models share one parameter
    tensor W of shape (M, d_max, n); rows >= widths[m] are zeroed by a
    fixed mask, so model m is exactly a width-widths[m] model (masked
    rows receive zero gradient through the mask multiplication).
    """

    def __init__(
        self,
        widths: list[int],
        n: int,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.widths = list(widths)
        self.n = n
        M, d_max = len(widths), max(widths)
        a = (6.0 / (n + d_max)) ** 0.5  # Xavier-uniform bound
        W = torch.empty(M, d_max, n, device=device).uniform_(-a, a, generator=generator)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(torch.zeros(M, n, device=device))
        mask = torch.zeros(M, d_max, 1, device=device)
        for m, d in enumerate(self.widths):
            mask[m, :d] = 1.0
        self.register_buffer("mask", mask)

    @property
    def W_eff(self) -> torch.Tensor:
        """(M, d_max, n) weights with masked rows exactly zero."""
        return self.W * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (M, B, n) per-model batches, or (B, n) shared across models."""
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(len(self.widths), -1, -1)
        W = self.W_eff
        h = torch.einsum("mdn,mbn->mbd", W, x)
        return torch.relu(torch.einsum("mdn,mbd->mbn", W, h) + self.b.unsqueeze(1))
