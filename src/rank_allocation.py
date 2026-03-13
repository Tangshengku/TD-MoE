from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass
class RankSearchConfig:
    target_reduction: float
    r1_list: Optional[Iterable[int]] = None
    r2_list: Optional[Iterable[int]] = None
    max_r1: Optional[int] = None
    max_r2: Optional[int] = None
    max_r3: Optional[int] = None
    step_r1: int = 1
    step_r2: int = 1


@dataclass
class RankSearchResult:
    r1: int
    r2: int
    r3: int
    params: int
    target_params: int
    diff: int


def _params_tucker(k: int, d_out: int, d_in: int, r1: int, r2: int, r3: int) -> int:
    return r1 * r2 * r3 + k * r1 + d_out * r2 + d_in * r3


def search_ranks(k: int, d_out: int, d_in: int, cfg: RankSearchConfig) -> RankSearchResult:
    if not 0.0 < cfg.target_reduction < 1.0:
        raise ValueError("target_reduction must be in (0, 1)")

    max_r1 = min(cfg.max_r1 or k, k)
    max_r2 = min(cfg.max_r2 or d_out, d_out)
    max_r3 = min(cfg.max_r3 or d_in, d_in)

    target_params = math.ceil((1.0 - cfg.target_reduction) * k * d_out * d_in)

    if cfg.r1_list is None:
        r1_list = range(1, max_r1 + 1, max(cfg.step_r1, 1))
    else:
        r1_list = [r for r in cfg.r1_list if 1 <= r <= max_r1]

    if cfg.r2_list is None:
        r2_list = range(1, max_r2 + 1, max(cfg.step_r2, 1))
    else:
        r2_list = [r for r in cfg.r2_list if 1 <= r <= max_r2]

    best = None
    for r1 in r1_list:
        for r2 in r2_list:
            # Derived from Eq. (5): r3 = (target_params - K*r1 - d_out*r2) / (r1*r2 + d_in)
            denom = r1 * r2 + d_in
            if denom <= 0:
                continue
            r3_float = (target_params - k * r1 - d_out * r2) / denom
            r3 = int(round(r3_float))
            r3 = max(1, min(max_r3, r3))

            params = _params_tucker(k, d_out, d_in, r1, r2, r3)
            diff = abs(params - target_params)
            if best is None or diff < best.diff:
                best = RankSearchResult(r1=r1, r2=r2, r3=r3, params=params, target_params=target_params, diff=diff)

    if best is None:
        raise RuntimeError("No feasible ranks found")
    return best
