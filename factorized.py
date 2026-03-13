from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedExpertLinear(nn.Module):
    """Linear layer for MoE experts using shared Tucker factors.

    Weight for expert i: W_i = sum_a U1[i,a] * (U2 @ G[a] @ U3^T)
    """

    def __init__(self, u1: torch.Tensor, u2: torch.Tensor, u3: torch.Tensor, core: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("u1", u1)
        self.register_buffer("u2", u2)
        self.register_buffer("u3", u3)
        self.register_buffer("core", core)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

        # Precompute H_a = U2 @ core[a,:,:] @ U3^T for all a
        # core shape: (r1, r2, r3)
        r1 = core.shape[0]
        h_list = []
        for a in range(r1):
            h = u2 @ core[a] @ u3.t()
            h_list.append(h)
        h = torch.stack(h_list, dim=0)
        self.register_buffer("h", h)

    def forward(self, x: torch.Tensor, expert_index: int):
        # x: (batch, d_in)
        coeff = self.u1[expert_index]  # (r1,)
        # Weighted sum of H_a @ x^T
        # Compute (r1, d_out) = (r1, d_out, d_in) @ (d_in, batch)
        hx = torch.einsum("aoi,bi->bao", self.h, x)
        y = torch.einsum("a,bao->bo", coeff, hx)
        if self.bias is not None:
            y = y + self.bias
        return y
