from __future__ import annotations

import torch
import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker

# Use PyTorch as the TensorLy backend throughout this module.
tl.set_backend("pytorch")


def whiten_tensor(tensor: torch.Tensor, s_out: torch.Tensor | None, s_in: torch.Tensor | None) -> torch.Tensor:
    """Apply multi-linear whitening: Tw = T ×2 Sout ×3 Sin (paper Eq. 4)."""
    # Whitening matrices are float32; TensorLy's mode_dot requires matching dtypes.
    if (s_out is not None or s_in is not None) and tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    out = tensor
    if s_out is not None:
        out = tl.tenalg.mode_dot(out, s_out, mode=1)  # output mode (dout)
    if s_in is not None:
        out = tl.tenalg.mode_dot(out, s_in, mode=2)   # input  mode (din)
    return out


def recolor_factors(factors, s_out_inv: torch.Tensor | None, s_in_inv: torch.Tensor | None):
    """Absorb inverse whitening into factors: U'2 = Sout^{-1} U2, U'3 = Sin^{-1} U3."""
    u1, u2, u3 = factors
    if s_out_inv is not None:
        u2 = s_out_inv @ u2
    if s_in_inv is not None:
        u3 = s_in_inv @ u3
    return [u1, u2, u3]


def tucker_decompose(
    tensor: torch.Tensor,
    ranks: tuple[int, int, int],
    init: str = "svd",
    tol: float = 1e-6,
    n_iter_max: int = 50,
    device_override: str | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Tucker decomposition via TensorLy (HOOI).  Returns (core, [U1, U2, U3])."""
    orig_dtype = tensor.dtype
    orig_device = tensor.device
    if not torch.isfinite(tensor).all():
        print("some infinite in the tensor")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    if device_override == "cpu":
        tensor = tensor.cpu()

    def _run(t: torch.Tensor):
        result = tl_tucker(t, rank=list(ranks), n_iter_max=n_iter_max, init=init, tol=tol)
        # TensorLy >= 0.7 returns a TuckerTensor named-tuple; older versions return a plain tuple.
        core = result.core if hasattr(result, "core") else result[0]
        factors = list(result.factors) if hasattr(result, "factors") else list(result[1])
        return core, factors

    try:
        core, factors = _run(tensor)
    except Exception as err:
        # CPU fallback for numerical stability (e.g. cusolver/magma errors on CUDA).
        if tensor.is_cuda and device_override != "cpu":
            core, factors = _run(tensor.float().cpu())
            core = core.to(orig_device)
            factors = [f.to(orig_device) for f in factors]
        else:
            raise err

    if orig_dtype != tensor.dtype:
        core = core.to(orig_dtype)
        factors = [f.to(orig_dtype) for f in factors]
    return core, factors


def reconstruct(core: torch.Tensor, factors) -> torch.Tensor:
    """Reconstruct full tensor from Tucker factors: G ×1 U1 ×2 U2 ×3 U3."""
    return tl.tucker_to_tensor((core, factors))
