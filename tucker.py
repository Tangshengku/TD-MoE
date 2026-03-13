from __future__ import annotations

import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import mode_dot
import torch


tl.set_backend("pytorch")


def whiten_tensor(tensor: torch.Tensor, s_out: torch.Tensor | None, s_in: torch.Tensor | None):
    out = tensor
    if s_out is not None:
        out = mode_dot(out, s_out, mode=1)
    if s_in is not None:
        out = mode_dot(out, s_in, mode=2)
    return out


def recolor_factors(factors, s_out_inv: torch.Tensor | None, s_in_inv: torch.Tensor | None):
    u1, u2, u3 = factors
    if s_out_inv is not None:
        u2 = s_out_inv @ u2
    if s_in_inv is not None:
        u3 = s_in_inv @ u3
    return [u1, u2, u3]


def tucker_decompose(tensor: torch.Tensor, ranks: tuple[int, int, int], init="svd", tol=1e-6, n_iter_max=50):
    core, factors = tucker(tensor, ranks=ranks, init=init, tol=tol, n_iter_max=n_iter_max)
    return core, factors


def reconstruct(core: torch.Tensor, factors):
    return tl.tucker_to_tensor((core, factors))
