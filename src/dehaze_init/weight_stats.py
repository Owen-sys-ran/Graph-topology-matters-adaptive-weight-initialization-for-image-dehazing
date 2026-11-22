"""Weight-statistics extraction for pretrained dehazing checkpoints."""

from __future__ import annotations

import re
from typing import Dict, List

import torch


def _fan(shape):
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return shape[0], shape[0]
    out_ch, in_ch = shape[0], shape[1]
    rec = 1
    for d in shape[2:]:
        rec *= d
    return in_ch * rec, out_ch * rec


def _role(name: str) -> str:
    n = name.lower()
    if "norm" in n:
        return "norm"
    if "qk" in n:
        return "qk"
    if "ca." in n or "pa." in n or "calayer" in n or "palayer" in n:
        return "gate"
    if "bias" in n:
        return "bias"
    if re.search(r"(head|tail|recon|post|out)", n):
        return "head"
    return "conv"


def _spectral_norm(w: torch.Tensor, iters: int = 6) -> float:
    if w.ndim < 2:
        return float(w.abs().max())
    a = w.reshape(w.shape[0], -1)
    v = torch.randn(a.shape[1], device=w.device, dtype=w.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        u = a @ v
        u = u / (u.norm() + 1e-12)
        v = a.t() @ u
        v = v / (v.norm() + 1e-12)
    return float((u @ a @ v).norm() / (u.norm() * v.norm() + 1e-12))


def tensor_stats(name: str, t: torch.Tensor) -> Dict:
    t = t.detach().cpu().float()
    fan_in, fan_out = _fan(t.shape)
    row = {
        "name": name,
        "role": _role(name),
        "shape": "x".join(str(d) for d in t.shape),
        "numel": t.numel(),
        "fan_in": fan_in,
        "fan_out": fan_out,
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "abs_mean": float(t.abs().mean()),
        "min": float(t.min()),
        "max": float(t.max()),
        "sparsity": float((t.abs() < 1e-6).float().mean()),
    }
    if t.ndim >= 2 and t.numel() > 1:
        row["spectral_norm"] = _spectral_norm(t)
        # Theory reference standard deviations for comparison.
        row["xavier_std"] = (2.0 / (fan_in + fan_out)) ** 0.5
        row["kaiming_std"] = (2.0 / fan_in) ** 0.5
    else:
        row["spectral_norm"] = float("nan")
        row["xavier_std"] = float("nan")
        row["kaiming_std"] = float("nan")
    return row


def analyze_state_dict(sd: Dict[str, torch.Tensor]) -> List[Dict]:
    rows: List[Dict] = []
    for name, t in sd.items():
        if not torch.is_tensor(t):
            continue
        rows.append(tensor_stats(name, t))
    return rows


def strip_ddp(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.removeprefix("module."): v for k, v in state_dict.items()}
