from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch


@dataclass
class ExpertGroup:
    name: str
    module: torch.nn.Module
    experts: List[torch.nn.Module]
    gate: torch.nn.Module | None = None


def find_expert_groups(model: torch.nn.Module) -> List[ExpertGroup]:
    groups = []
    for name, module in model.named_modules():
        if hasattr(module, "experts"):
            experts = getattr(module, "experts")
            if isinstance(experts, torch.nn.ModuleList) and len(experts) > 0:
                gate = getattr(module, "gate", None)
                groups.append(ExpertGroup(name=name, module=module, experts=list(experts), gate=gate))
    return groups


def get_expert_linear_modules(expert: torch.nn.Module, linear_names: Iterable[str]) -> Dict[str, torch.nn.Linear]:
    out = {}
    for name in linear_names:
        if hasattr(expert, name):
            mod = getattr(expert, name)
            if isinstance(mod, torch.nn.Linear):
                out[name] = mod
    return out


def stack_expert_weights(experts: List[torch.nn.Module], linear_name: str) -> torch.Tensor:
    weights = []
    for exp in experts:
        if not hasattr(exp, linear_name):
            raise ValueError(f"Expert missing linear {linear_name}")
        mod = getattr(exp, linear_name)
        if not isinstance(mod, torch.nn.Linear):
            raise ValueError(f"Expert attribute {linear_name} is not Linear")
        weights.append(mod.weight.data)
    return torch.stack(weights, dim=0)


def apply_expert_weights(experts: List[torch.nn.Module], linear_name: str, weights: torch.Tensor):
    if weights.dim() != 3:
        raise ValueError("weights must be (K, d_out, d_in)")
    if len(experts) != weights.shape[0]:
        raise ValueError("Expert count does not match weights")
    for i, exp in enumerate(experts):
        mod = getattr(exp, linear_name)
        mod.weight.data.copy_(weights[i])
