from __future__ import annotations

import time
import torch
import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker

# Use PyTorch as the TensorLy backend throughout this module.
tl.set_backend("pytorch")


def whiten_tensor(tensor: torch.Tensor, s_out: torch.Tensor | None, s_in: torch.Tensor | None) -> torch.Tensor:
    """Apply multi-linear whitening: Tw = T ×2 Sout ×3 Sin (paper Eq. 4)."""
    print(f"[whiten_tensor] input shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
          f"s_out={'None' if s_out is None else tuple(s_out.shape)} "
          f"s_in={'None' if s_in is None else tuple(s_in.shape)}")
    # Whitening matrices are float32; TensorLy's mode_dot requires matching dtypes.
    if (s_out is not None or s_in is not None) and tensor.dtype in (torch.float16, torch.bfloat16):
        print(f"[whiten_tensor] upcasting tensor {tensor.dtype} -> float32 for mode_dot")
        tensor = tensor.float()
    out = tensor
    if s_out is not None:
        out = tl.tenalg.mode_dot(out, s_out, mode=1)  # output mode (dout)
        print(f"[whiten_tensor] after ×2 Sout: shape={tuple(out.shape)} "
              f"finite={torch.isfinite(out).all().item()} "
              f"abs_max={out.abs().max().item():.4g}")
    if s_in is not None:
        out = tl.tenalg.mode_dot(out, s_in, mode=2)   # input  mode (din)
        print(f"[whiten_tensor] after ×3 Sin:  shape={tuple(out.shape)} "
              f"finite={torch.isfinite(out).all().item()} "
              f"abs_max={out.abs().max().item():.4g}")
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

    print(f"[tucker_decompose] shape={tuple(tensor.shape)} dtype={orig_dtype} device={orig_device} "
          f"ranks={ranks} device_override={device_override!r}")

    finite_before = torch.isfinite(tensor).all().item()
    print(f"[tucker_decompose] input finite={finite_before} "
          f"abs_max={tensor.float().abs().max().item():.4g} "
          f"abs_min_nonzero={tensor.float().abs()[tensor != 0].min().item():.4g}")

    if not finite_before:
        print("[tucker_decompose] WARNING: non-finite values detected — applying nan_to_num")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)

    if tensor.dtype in (torch.float16, torch.bfloat16):
        print(f"[tucker_decompose] upcasting {tensor.dtype} -> float32")
        tensor = tensor.float()

    if device_override == "cpu":
        print("[tucker_decompose] moving tensor to CPU (device_override='cpu')")
        tensor = tensor.cpu()

    def _run(t: torch.Tensor, init_mode: str = init):
        t0 = time.time()
        print(f"[tucker_decompose._run] device={t.device} dtype={t.dtype} "
              f"shape={tuple(t.shape)} init={init_mode!r} ranks={list(ranks)}")
        result = tl_tucker(t, rank=list(ranks), n_iter_max=n_iter_max, init=init_mode, tol=tol)
        # TensorLy >= 0.7 returns a TuckerTensor named-tuple; older versions return a plain tuple.
        core = result.core if hasattr(result, "core") else result[0]
        factors = list(result.factors) if hasattr(result, "factors") else list(result[1])
        elapsed = time.time() - t0
        print(f"[tucker_decompose._run] done in {elapsed:.1f}s  "
              f"core={tuple(core.shape)} "
              f"factor_shapes={[tuple(f.shape) for f in factors]}")
        return core, factors

    try:
        core, factors = _run(tensor)
    except Exception as cuda_err:
        if not tensor.is_cuda or device_override == "cpu":
            # Already on CPU; SVD init failed — retry with random init.
            print(f"[tucker_decompose] CPU SVD init failed ({type(cuda_err).__name__}: {cuda_err}); "
                  f"retrying with init='random'")
            t_clean = torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
            core, factors = _run(t_clean, init_mode="random")
        else:
            # CPU fallback for CUDA numerical errors (cusolver / magma SVD failures).
            print(f"[tucker_decompose] CUDA failed ({type(cuda_err).__name__}: {cuda_err}); "
                  f"retrying on CPU with nan_to_num cleanup")
            t_cpu = torch.nan_to_num(tensor.float().cpu(), nan=0.0, posinf=1e4, neginf=-1e4)
            print(f"[tucker_decompose] CPU tensor: finite={torch.isfinite(t_cpu).all().item()} "
                  f"abs_max={t_cpu.abs().max().item():.4g}")
            try:
                core, factors = _run(t_cpu)
            except Exception as cpu_err:
                # SVD init also failed on CPU; last resort: random initialization.
                print(f"[tucker_decompose] CPU SVD also failed ({type(cpu_err).__name__}: {cpu_err}); "
                      f"last resort: init='random' on CPU")
                core, factors = _run(t_cpu, init_mode="random")
            core = core.to(orig_device)
            factors = [f.to(orig_device) for f in factors]

    if orig_dtype != tensor.dtype:
        print(f"[tucker_decompose] casting output back to {orig_dtype}")
        core = core.to(orig_dtype)
        factors = [f.to(orig_dtype) for f in factors]

    print(f"[tucker_decompose] complete — core={tuple(core.shape)} dtype={core.dtype} device={core.device}")
    return core, factors


def reconstruct(core: torch.Tensor, factors) -> torch.Tensor:
    """Reconstruct full tensor from Tucker factors: G ×1 U1 ×2 U2 ×3 U3."""
    return tl.tucker_to_tensor((core, factors))
