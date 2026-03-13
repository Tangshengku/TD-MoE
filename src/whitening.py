from __future__ import annotations

import torch


def covariance_from_batches(batches, eps: float = 1e-6):
    """Compute covariance matrix from an iterable of 2D tensors (N, D).

    Returns (cov, count).
    """
    device = None
    dtype = None
    sum_xxt = None
    count = 0
    for x in batches:
        if x is None:
            continue
        if x.dim() != 2:
            x = x.view(-1, x.shape[-1])
        if device is None:
            device = x.device
            dtype = x.dtype
        if sum_xxt is None:
            sum_xxt = torch.zeros(x.shape[1], x.shape[1], device=device, dtype=dtype)
        x = x.float()
        sum_xxt += x.t() @ x
        count += x.shape[0]
    if sum_xxt is None:
        raise ValueError("No batches provided for covariance computation")
    cov = sum_xxt / max(count, 1)
    cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return cov, count


def whitening_matrix(cov: torch.Tensor):
    """Return whitening matrix S = cov^{-1/2} and its inverse S_inv = cov^{1/2}."""
    # Symmetric eigendecomposition
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=1e-12)
    evals_inv_sqrt = torch.rsqrt(evals)
    evals_sqrt = torch.sqrt(evals)
    s = (evecs * evals_inv_sqrt) @ evecs.t()
    s_inv = (evecs * evals_sqrt) @ evecs.t()
    return s, s_inv


def compute_whitening_from_activations(activations, eps: float = 1e-6):
    cov, _ = covariance_from_batches(activations, eps=eps)
    return whitening_matrix(cov)
