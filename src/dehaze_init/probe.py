"""Init-time forward / backward probe for a dehazing network.

Measures signal and gradient flow at initialization (no optimizer step):
  * per-layer activation standard deviation and mean absolute magnitude,
  * output range, identity residual (||out - I||), PSNR against the clean pair,
  * gradient norm, per-stage gradient std, and share of zero gradients.
"""

from __future__ import annotations

import re
from typing import Dict, List

import torch
import torch.nn as nn


def _stage_of(name: str) -> str:
    n = name.lower()
    if "stem" in n or "pre" in n or "e_conv1" in n:
        return "stem"
    if "head" in n or "post" in n or "e_conv5" in n or "recon" in n:
        return "head"
    if "block" in n or "group" in n or "e_conv" in n or "layer" in n:
        return "body"
    return "other"


def psnr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse < eps:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)))


@torch.no_grad()
def _forward_capture(model: nn.Module, x: torch.Tensor) -> Dict:
    acts: List[tuple] = []
    hooks = []

    def hook(name):
        def _h(_m, _i, out):
            o = out.detach()
            acts.append((name, float(o.std()), float(o.abs().mean())))
        return _h

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook(name)))

    out = model(x)
    for h in hooks:
        h.remove()

    layer_std = [a[1] for a in acts]
    return {
        "out": out,
        "layer_act_std": layer_std,
        "layer_names": [a[0] for a in acts],
        "acts": acts,
    }


def measure(model: nn.Module, x: torch.Tensor, clean: torch.Tensor,
            track_grad: bool = True) -> Dict:
    model.eval()
    cap = _forward_capture(model, x)
    out = cap["out"]

    res = {
        "in_std": float(x.std()),
        "out_mean": float(out.mean()),
        "out_std": float(out.std()),
        "out_min": float(out.min()),
        "out_max": float(out.max()),
        "identity_residual": float(torch.norm(out - x) / (torch.norm(x) + 1e-12)),
        "psnr_vs_clean": psnr(out, clean),
        "act_std_first": cap["layer_act_std"][0] if cap["layer_act_std"] else float("nan"),
        "act_std_last": cap["layer_act_std"][-1] if cap["layer_act_std"] else float("nan"),
        "act_std_min": float(min(cap["layer_act_std"])) if cap["layer_act_std"] else float("nan"),
        "act_std_max": float(max(cap["layer_act_std"])) if cap["layer_act_std"] else float("nan"),
        "n_layers": len(cap["layer_act_std"]),
    }

    if track_grad:
        model.train()
        model.zero_grad(set_to_none=True)
        out_g = model(x)
        loss = torch.nn.functional.mse_loss(out_g, clean)
        loss.backward()

        grad_by_stage: Dict[str, List[float]] = {"stem": [], "body": [], "head": [], "other": []}
        zero_grads = 0
        total_grads = 0
        all_std = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().cpu().float()
            total_grads += p.numel()
            zero_grads += int((g.abs() < 1e-8).sum().item())
            std = float(g.std(unbiased=False))
            all_std.append(std)
            grad_by_stage[_stage_of(name)].append(std)

        def _mean(lst):
            return float(sum(lst) / len(lst)) if lst else float("nan")

        res.update({
            "loss_at_init": float(loss.item()),
            "grad_norm": float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.grad is not None], float("inf"))),
            "grad_std_mean": _mean(all_std),
            "grad_zero_frac": zero_grads / max(total_grads, 1),
            "grad_std_stem": _mean(grad_by_stage["stem"]),
            "grad_std_body": _mean(grad_by_stage["body"]),
            "grad_std_head": _mean(grad_by_stage["head"]),
        })
        model.eval()

    return res
