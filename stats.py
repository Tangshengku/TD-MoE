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
        self.sum_xxt = torch.zeros(self.d, self.d, device=self.device, dtype=self.dtype)
        self.count = 0

    def update(self, x: torch.Tensor):
        if x is None:
            return
        if x.dim() != 2:
            x = x.view(-1, x.shape[-1])
        self.sum_xxt += x.t() @ x
        self.count += x.shape[0]

    def finalize(self):
        cov = self.sum_xxt / max(self.count, 1)
        cov = cov + self.eps * torch.eye(self.d, device=self.device, dtype=self.dtype)
        return cov, self.count


def whitening_from_cov(cov: torch.Tensor):
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=1e-12)
    evals_inv_sqrt = torch.rsqrt(evals)
    evals_sqrt = torch.sqrt(evals)
    s = (evecs * evals_inv_sqrt) @ evecs.t()
    s_inv = (evecs * evals_sqrt) @ evecs.t()
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
