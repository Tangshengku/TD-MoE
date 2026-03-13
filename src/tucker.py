from __future__ import annotations

import torch


def _unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    order = (mode,) + tuple(i for i in range(tensor.ndim) if i != mode)
    return tensor.permute(order).reshape(tensor.shape[mode], -1)


def mode_dot(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("mode_dot expects a 2D matrix")
    if tensor.shape[mode] != matrix.shape[1]:
        raise ValueError("mode_dot matrix has incompatible shape")
    order = (mode,) + tuple(i for i in range(tensor.ndim) if i != mode)
    transposed = tensor.permute(order)
    unfolded = transposed.reshape(tensor.shape[mode], -1)
    product = matrix @ unfolded
    new_shape = (matrix.shape[0],) + tuple(tensor.shape[i] for i in range(tensor.ndim) if i != mode)
    folded = product.reshape(new_shape)
    inv_order = tuple(order.index(i) for i in range(tensor.ndim))
    return folded.permute(inv_order)


def tucker_to_tensor(tucker_tensor) -> torch.Tensor:
    core, factors = tucker_tensor
    out = core
    for mode, factor in enumerate(factors):
        out = mode_dot(out, factor, mode)
    return out


def tucker(
    tensor: torch.Tensor,
    rank: tuple[int, ...],
    init: str = "svd",
    tol: float = 1e-6,
    n_iter_max: int = 50,
):
    if len(rank) != tensor.ndim:
        raise ValueError("rank must match tensor.ndim")
    if init not in {"svd", "random"}:
        raise ValueError(f"Unsupported init: {init}")

    factors = []
    if init == "svd":
        for mode in range(tensor.ndim):
            unfolded = _unfold(tensor, mode)
            u, _s, _vh = torch.linalg.svd(unfolded, full_matrices=False)
            factors.append(u[:, : rank[mode]])
    else:
        for mode in range(tensor.ndim):
            mat = torch.randn(
                tensor.shape[mode],
                rank[mode],
                device=tensor.device,
                dtype=tensor.dtype,
            )
            q, _r = torch.linalg.qr(mat, mode="reduced")
            factors.append(q)

    norm_tensor = torch.linalg.norm(tensor)
    last_error = None
    for _ in range(n_iter_max):
        for mode in range(tensor.ndim):
            core = tensor
            for m in range(tensor.ndim):
                if m == mode:
                    continue
                core = mode_dot(core, factors[m].T, m)
            unfolded = _unfold(core, mode)
            u, _s, _vh = torch.linalg.svd(unfolded, full_matrices=False)
            factors[mode] = u[:, : rank[mode]]

        core = tensor
        for m in range(tensor.ndim):
            core = mode_dot(core, factors[m].T, m)
        rec = tucker_to_tensor((core, factors))
        error = torch.linalg.norm(tensor - rec) / (norm_tensor + 1e-12)
        if last_error is not None and abs(last_error - error) < tol:
            break
        last_error = error

    return core, factors


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


def tucker_decompose(
    tensor: torch.Tensor,
    ranks: tuple[int, int, int],
    init="svd",
    tol=1e-6,
    n_iter_max=50,
    device_override: str | None = None,
):
    orig_dtype = tensor.dtype
    orig_device = tensor.device
    if not torch.isfinite(tensor).all():
        print("some infinite in the tensor")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    if device_override == "cpu":
        tensor = tensor.cpu()
    try:
        try:
            core, factors = tucker(tensor, rank=ranks, init=init, tol=tol, n_iter_max=n_iter_max)
        except TypeError:
            core, factors = tucker(tensor, ranks=ranks, init=init, tol=tol, n_iter_max=n_iter_max)
    except BaseException as err:
        # CPU fallback for stability (e.g., cusolver/magma errors)
        if tensor.is_cuda and device_override != "cpu":
            tensor_cpu = tensor.float().cpu()
            try:
                core, factors = tucker(tensor_cpu, rank=ranks, init=init, tol=tol, n_iter_max=n_iter_max)
            except TypeError:
                core, factors = tucker(tensor_cpu, ranks=ranks, init=init, tol=tol, n_iter_max=n_iter_max)
            core = core.to(orig_device)
            factors = [f.to(orig_device) for f in factors]
        else:
            raise err
    if orig_dtype != tensor.dtype:
        core = core.to(orig_dtype)
        factors = [f.to(orig_dtype) for f in factors]
    return core, factors


def reconstruct(core: torch.Tensor, factors):
    return tucker_to_tensor((core, factors))
