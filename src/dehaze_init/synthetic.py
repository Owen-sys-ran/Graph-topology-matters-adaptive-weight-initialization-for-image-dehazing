"""Synthetic hazy pairs built directly from the atmospheric scattering model.

No external dataset is required. A clean patch ``J`` is synthesized, then the
observed hazy image is formed as::

    I(x) = J(x) * t(x) + A * (1 - t(x))

with per-image atmospheric light ``A`` and a smooth depth-derived transmission
map ``t(x)``.
"""

from __future__ import annotations

import torch


def _smooth_map(b: int, c: int, h: int, w: int, low: float, high: float,
                generator: torch.Generator) -> torch.Tensor:
    """Random smooth field via upsampled low-resolution noise."""
    small = torch.rand(b, c, max(h // 8, 1), max(w // 8, 1), generator=generator)
    up = torch.nn.functional.interpolate(small, size=(h, w), mode="bilinear",
                                        align_corners=False)
    return low + (high - low) * up


def make_pair(batch: int = 4, size: int = 64, seed: int = 0,
              device: str = "cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    h = w = size

    # Clean radiance: smooth colour gradient + fine texture, in [0, 1].
    base = _smooth_map(batch, 3, h, w, 0.15, 0.85, g)
    texture = torch.randn(batch, 3, h, w, generator=g) * 0.08
    j = torch.clamp(base + texture, 0.0, 1.0)

    # Atmospheric light A in [0.7, 1.0]; transmission t in [0.3, 1.0].
    a = _smooth_map(batch, 3, h, w, 0.7, 1.0, g)
    t = _smooth_map(batch, 1, h, w, 0.3, 1.0, g)
    t = t.expand_as(j)

    i = j * t + a * (1.0 - t)
    i = torch.clamp(i, 0.0, 1.0)
    return i.to(device), j.to(device), a.to(device), t.to(device)


def make_pool(n: int, size: int = 32, seed: int = 0, device: str = "cpu"):
    """Stack `n` synthetic hazy/clean pairs into fixed tensor pools."""
    imgs, js = [], []
    for k in range(n):
        i, j, _, _ = make_pair(1, size, seed=seed + k, device=device)
        imgs.append(i[0])
        js.append(j[0])
    return torch.stack(imgs).to(device), torch.stack(js).to(device)
