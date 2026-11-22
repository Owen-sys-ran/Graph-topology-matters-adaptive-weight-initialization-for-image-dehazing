"""
Parameter initialization schemes for image-dehazing networks.

This module collects:
  * classic baselines used by public dehazing repos
    (PyTorch default, Xavier/Glorot, He/Kaiming, DCGAN-style normal(0, 0.02),
     DehazeFormer-style depth-scaled truncated normal), and
  * DPI -- Dehazing-aware Parameter Initialization, the proposed algorithm.

All routines operate on ``nn.Module`` in place and are CPU friendly.

Conventions
-----------
fan_in  = number of input connections  (in_channels * k_h * k_w for conv)
fan_out = number of output connections (out_channels * k_h * k_w for conv)

Role tags recognized by :func:`dpi_init` (matched against parameter/module name):
  * ``qk``      : query/key projection (symmetric, no depth gain)
  * ``gate``    : final conv feeding a sigmoid attention gate
  * ``residual``: last conv of a residual branch / reconstruction head
  * ``norm``    : normalization layer (weight=1, bias=0)
  * ``head``    : explicit transmission / atmospheric-light prediction head
"""

from __future__ import annotations

import math
import re
from typing import Dict, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fan_in_out(weight: torch.Tensor):
    if weight.ndim < 2:
        return weight.numel(), weight.numel()
    out_ch, in_ch = weight.shape[:2]
    receptive = weight[0][0].numel()
    fan_in = in_ch * receptive
    fan_out = out_ch * receptive
    return fan_in, fan_out


def _is_norm(name: str, module: nn.Module) -> bool:
    return isinstance(
        module,
        (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm, nn.GroupNorm),
    ) or "norm" in name.lower()


def _role(name: str) -> str:
    n = name.lower()
    if "norm" in n:
        return "norm"
    if "qk" in n or "q_conv" in n or "q_linear" in n:
        return "qk"
    if "gate" in n or "palayer" in n or "calayer" in n or "pa." in n or "ca." in n:
        return "gate"
    if "residual" in n or "tail" in n or "recon" in n or "head_out" in n or "post" in n:
        return "residual"
    if "trans" in n or "t_head" in n or "airlight" in n or "a_head" in n or "atm" in n:
        return "head"
    return "body"


# --------------------------------------------------------------------------- #
# classic baselines
# --------------------------------------------------------------------------- #
def default_init(_: nn.Module) -> None:
    """PyTorch native default: conv/linear use kaiming_uniform(a=sqrt5)."""
    return None  # modules already carry PyTorch default at construction


def xavier_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif _is_norm("", m):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def kaiming_init(module: nn.Module, neg_slope: float = 0.0) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=neg_slope, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif _is_norm("", m):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def dcgan_init(module: nn.Module) -> None:
    """DCGAN-style recipe used by the AOD-Net PyTorch reproduction:
    conv weight ~ N(0, 0.02); BN weight ~ N(1, 0.02), bias = 0."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)


def depth_scaled_trunc(module: nn.Module, depth: int, std_floor: float = 0.02) -> None:
    """DehazeFormer-style: truncated normal with a depth-dependent gain.

    gain = (8 * D) ** -1/4 ; std = gain * sqrt(2 / (fan_in + fan_out)).
    QK projection uses plain Xavier truncation.
    """
    gain = (8.0 * max(depth, 1)) ** (-0.25)
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            role = _role(name)
            fan_in, fan_out = _fan_in_out(m.weight)
            if role == "qk":
                std = math.sqrt(2.0 / float(fan_in + fan_out))
            else:
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            std = max(std, std_floor * 0.1)
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif _is_norm(name, m):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------- #
# DPI: proposed dehazing-aware initialization
# --------------------------------------------------------------------------- #
def dpi_init(
    module: nn.Module,
    depth: int = 1,
    t0: float = 0.9,
    a0: float = 0.9,
    residual_zero: bool = True,
    gate_bias: float = 0.0,
    depth_gain: bool = True,
    head_std_scale: float = 0.1,
    seed: Optional[int] = None,
) -> None:
    """Dehazing-aware Parameter Initialization (DPI).

    Parameters
    ----------
    module : nn.Module
        Network to initialize in place.
    depth : int
        Total residual/block depth, used for the depth-scaled variance gain.
    t0 : float
        Prior mean transmission for an explicit transmission head
        (sigmoid output). Bias = logit(t0).
    a0 : float
        Prior atmospheric light for an explicit airlight head (sigmoid output).
    residual_zero : bool
        Zero-initialize the residual branch / reconstruction head so the
        network starts as an identity mapping (J ~= I), the standard
        restoration prior.
    gate_bias : float
        Bias for sigmoid attention gates. 0.0 -> gate starts at 0.5.
    depth_gain : bool
        Apply the (8D)^-1/4 depth gain to body/MLP projections.
    """
    if seed is not None:
        torch.manual_seed(seed)

    gain = (8.0 * max(depth, 1)) ** (-0.25) if depth_gain else 1.0
    eps = 1e-6

    def logit(p: float) -> float:
        p = min(max(p, eps), 1 - eps)
        return math.log(p / (1.0 - p))

    for name, m in module.named_modules():
        if _is_norm(name, m):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
            continue

        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue

        role = _role(name)
        fan_in, fan_out = _fan_in_out(m.weight)

        if role == "residual" and residual_zero:
            # Identity-start restoration head: branch contributes ~0 at init.
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            continue

        if role == "qk":
            std = math.sqrt(2.0 / float(fan_in + fan_out))
        elif role == "gate":
            # Attention gate: small He scale, bias sets initial gate response.
            std = math.sqrt(2.0 / float(fan_in))
        elif role == "residual":
            # Non-zero but small-scale head: stays close to identity while
            # keeping first-step gradient flowing into the body.
            std = head_std_scale * gain * math.sqrt(2.0 / float(fan_in))
        else:  # body / mlp / v / proj
            std = gain * math.sqrt(2.0 / float(fan_in))

        nn.init.trunc_normal_(m.weight, mean=0.0, std=max(std, 1e-5), a=-2 * std, b=2 * std)

        if m.bias is None:
            continue
        if role == "head":
            if "trans" in name.lower():
                nn.init.constant_(m.bias, logit(t0))
            elif "airlight" in name.lower() or "atm" in name.lower():
                nn.init.constant_(m.bias, logit(a0))
            else:
                nn.init.zeros_(m.bias)
        elif role == "gate":
            nn.init.constant_(m.bias, gate_bias)
        else:
            nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------- #
# GAI: Graph-Aware Initialization (proposed, graph-topology driven)
# --------------------------------------------------------------------------- #
def gai_init(
    module: nn.Module,
    depth: int = 1,
    x: Optional[torch.Tensor] = None,
    graph_info=None,
    t0: float = 0.9,
    a0: float = 0.9,
    residual_zero: bool = True,
    gate_bias: float = 0.0,
    head_std_scale: float = 0.1,
    seed: Optional[int] = None,
) -> None:
    """Graph-Aware Initialization (GAI).

    Instead of a single global depth gain, GAI first traces the network's
    computational DAG, then allocates each conv/linear node a variance budget
    from its graph position:

      * critical-path depth ``rho_v`` in [0, 1] scales the depth gain
        ``g_v = (8 (1 + rho_v (D-1)))^-1/4``: stem nodes keep the largest
        budget, nodes near the output on the longest residual path receive the
        smallest budget;
      * dense-branch count ``k_v`` (in-degree of a feeding concat node) scales
        the std by ``1/sqrt(k_v)`` so that k fused branches do not sum their
        second moments uncontrolled;
      * role rules are inherited from DPI: QK Xavier, attention gate small He,
        reconstruction head zero (identity start) or 0.1x small-scale,
        physics-head prior bias, norm weight 1 / bias 0.

    Parameters
    ----------
    module : nn.Module
    depth : int
        Residual/block depth reference ``D``.
    x : torch.Tensor, optional
        Dummy input used to trace the graph. A 1x3x32x32 tensor is built if
        omitted.
    graph_info : GraphInfo, optional
        Pre-computed :class:`dehaze_init.graph.GraphInfo`; traced on the fly
        otherwise.
    """
    from dehaze_init.graph import trace_graph

    if seed is not None:
        torch.manual_seed(seed)

    if graph_info is None:
        if x is None:
            x = torch.rand(1, 3, 32, 32)
        graph_info = trace_graph(module, x)

    D = max(depth, 1)
    d_crit = max(int(graph_info.summary["critical_depth"]), 1)
    eps = 1e-6

    def logit(p: float) -> float:
        p = min(max(p, eps), 1 - eps)
        return math.log(p / (1.0 - p))

    for name, m in module.named_modules():
        if _is_norm(name, m):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
            continue

        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue

        desc = graph_info.node_of(name)
        rho = float(desc.get("depth", 0)) / float(d_crit)
        k_v = max(int(desc.get("branch_in", 1)), 1)

        gain = (8.0 * (1.0 + rho * (D - 1))) ** (-0.25)
        branch_scale = 1.0 / math.sqrt(float(k_v))

        role = _role(name)
        fan_in, fan_out = _fan_in_out(m.weight)

        if role == "residual" and residual_zero:
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            continue

        if role == "qk":
            std = math.sqrt(2.0 / float(fan_in + fan_out))
        elif role == "gate":
            std = math.sqrt(2.0 / float(fan_in))
        elif role == "residual":
            std = head_std_scale * gain * branch_scale * math.sqrt(2.0 / float(fan_in))
        else:
            std = gain * branch_scale * math.sqrt(2.0 / float(fan_in))

        nn.init.trunc_normal_(m.weight, mean=0.0, std=max(std, 1e-5),
                              a=-2 * std, b=2 * std)

        if m.bias is None:
            continue
        if role == "head":
            if "trans" in name.lower():
                nn.init.constant_(m.bias, logit(t0))
            elif "airlight" in name.lower() or "atm" in name.lower():
                nn.init.constant_(m.bias, logit(a0))
            else:
                nn.init.zeros_(m.bias)
        elif role == "gate":
            nn.init.constant_(m.bias, gate_bias)
        else:
            nn.init.zeros_(m.bias)


def calibrate_forward(module: nn.Module, x: torch.Tensor, target_std: float = 1.0,
                      clip: float = 3.0) -> None:
    """Data-dependent variance calibration (optional DPI stage).

    Runs one forward pass, measures each conv/linear output std, and rescales
    the layer weight so its output second moment approaches ``target_std``.
    Bias is left untouched. This is a lightweight stand-in for the batch
    statistics absorbed by normalization layers.
    """
    acts: Dict[str, torch.Tensor] = {}
    hooks = []

    def _hook(name):
        def _h(_m, _i, out):
            acts[name] = out.detach()
        return _h

    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(_hook(name)))

    module.eval()
    with torch.no_grad():
        module(x)

    for h in hooks:
        h.remove()

    named = dict(module.named_modules())
    for name, out in acts.items():
        obs = float(out.std(unbiased=False))
        if obs < 1e-6:
            continue
        scale = min(max(target_std / obs, 1.0 / clip), clip)
        m = named[name]
        with torch.no_grad():
            m.weight.mul_(scale)


SCHEMES = {
    "default": default_init,
    "xavier": xavier_init,
    "kaiming": kaiming_init,
    "dcgan": dcgan_init,
    "depth_trunc": depth_scaled_trunc,
    "dpi": dpi_init,
    "gai": gai_init,
}
