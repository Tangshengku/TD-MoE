from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch


@dataclass
class OnlineCovariance:
    d: int
    device: torch.device
    dtype: torch.dtype
    eps: float = 1e-6

    def __post_init__(self):
        self.sum_xxt = torch.zeros(self.d, self.d, device=self.device, dtype=torch.float32)
        self.count = 0
        self.orig_dtype = torch.float16

    def update(self, x: torch.Tensor):
        if x is None:
            return
        if x.dim() != 2:
            x = x.view(-1, x.shape[-1])
        x = x.float()
        self.sum_xxt += x.t() @ x
        self.count += x.shape[0]

    def finalize(self):
        cov = self.sum_xxt / max(self.count, 1)
        cov = cov + self.eps * torch.eye(self.d, device=self.device, dtype=self.orig_dtype)
        cov = cov.to(self.orig_dtype)
        return cov, self.count


def whitening_from_cov(cov: torch.Tensor):
    # torch.linalg.eigh doesn't support CUDA fp16. Do eigendecomp in fp32.
    orig_dtype = cov.dtype
    device = cov.device
    if cov.dtype in (torch.float16, torch.bfloat16):
        cov = cov.float()

    # Ensure symmetry and add jitter if needed.
    cov = 0.5 * (cov + cov.t())
    if not torch.isfinite(cov).all():
        cov = torch.nan_to_num(cov, nan=0.0, posinf=1e4, neginf=-1e4)

    def _eigh(mat: torch.Tensor):
        return torch.linalg.eigh(mat)

    # Try with increasing jitter, fallback to CPU float64 if needed.
    jitter_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    last_err = None
    for jitter in jitter_list:
        try:
            if jitter > 0:
                mat = mat = cov + jitter * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            else:
                mat = cov
            evals, evecs = _eigh(mat)
            break
        except RuntimeError as err:
            last_err = err
            evals = evecs = None
    else:
        evals = evecs = None

    if evals is None or evecs is None:
        # CPU float64 fallback for numerical stability
        cov_cpu = cov.double().cpu()
        evals, evecs = torch.linalg.eigh(cov_cpu)
        evals = evals.to(cov.device, dtype=cov.dtype)
        evecs = evecs.to(cov.device, dtype=cov.dtype)

    evals = torch.clamp(evals, min=1e-12)
    evals_inv_sqrt = torch.rsqrt(evals)
    evals_sqrt = torch.sqrt(evals)
    s = (evecs * evals_inv_sqrt) @ evecs.t()
    s_inv = (evecs * evals_sqrt) @ evecs.t()
    s = torch.nan_to_num(s, nan=0.0, posinf=1e4, neginf=-1e4)
    s_inv = torch.nan_to_num(s_inv, nan=0.0, posinf=1e4, neginf=-1e4)
    if orig_dtype != cov.dtype:
        s = s.to(orig_dtype)
        s_inv = s_inv.to(orig_dtype)
    return s, s_inv


class ActivationCollector:
    def __init__(self, module: torch.nn.Module, collect_grad: bool = False):
        self.module = module
        self.collect_grad = collect_grad
        self.inputs = []
        self.output_grads = []
        self.handles = []

    def _forward_hook(self, _mod, inp, out):
        x = inp[0].detach()
        self.inputs.append(x)
        if self.collect_grad:
            out.retain_grad()

    def _backward_hook(self, _mod, _grad_inp, grad_out):
        if not self.collect_grad:
            return
        if grad_out and grad_out[0] is not None:
            self.output_grads.append(grad_out[0].detach())

    def start(self):
        self.handles.append(self.module.register_forward_hook(self._forward_hook))
        if self.collect_grad:
            self.handles.append(self.module.register_full_backward_hook(self._backward_hook))

    def stop(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self):
        self.inputs.clear()
        self.output_grads.clear()
