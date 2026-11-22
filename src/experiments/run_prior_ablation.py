"""Physics-head prior-bias ablation.

A tiny network explicitly predicts transmission ``t`` and atmospheric light
``A`` and inverts the ASM::

    J = (I - A * (1 - t)) / t

We sweep the prior biases ``t0`` (transmission) and ``a0`` (airlight) written
into the sigmoid-head biases, and measure init-time reconstruction error
against the clean image. This isolates the physics-aware bias component of DPI.

Outputs
-------
analysis/prior_ablation.csv
analysis/figs/prior_ablation.png
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dehaze_init import schemes, synthetic, probe  # noqa: E402


class ASMHeadNet(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.stem = nn.Conv2d(3, dim, 3, padding=1)
        self.body = nn.Conv2d(dim, dim, 3, padding=1)
        self.t_head = nn.Conv2d(dim, 1, 3, padding=1)
        self.a_head = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x):
        f = torch.relu(self.body(torch.relu(self.stem(x))))
        t = torch.sigmoid(self.t_head(f))
        a = torch.sigmoid(self.a_head(f))
        a = a.mean(dim=(2, 3), keepdim=True)  # global atmospheric light
        t = t.clamp(1e-3, 1.0)
        j = (x - a * (1.0 - t)) / t
        return j.clamp(0.0, 1.0), t, a


def logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def main() -> None:
    os.makedirs(os.path.join(ROOT, "analysis", "figs"), exist_ok=True)
    x, j, A, t = synthetic.make_pair(batch=4, size=64, seed=1)

    rows = []
    t0_grid = [0.5, 0.7, 0.9, 0.97]
    a0_grid = [0.8, 0.9, 0.97]

    for t0 in t0_grid:
        for a0 in a0_grid:
            torch.manual_seed(0)
            net = ASMHeadNet()
            schemes.dpi_init(net, depth=2, residual_zero=False)
            # Overwrite physics-head biases with the chosen priors.
            with torch.no_grad():
                net.t_head.bias.fill_(logit(t0))
                net.a_head.bias.fill_(logit(a0))

            out, t_hat, a_hat = net(x)
            out = out.detach()
            rows.append({
                "t0": t0,
                "a0": a0,
                "psnr": probe.psnr(out, j),
                "mse": float(torch.mean((out - j) ** 2)),
                "t_mean": float(t_hat.detach().mean()),
                "a_mean": float(a_hat.detach().mean()),
                "out_mean": float(out.mean()),
            })
            print(f"t0={t0:.2f} a0={a0:.2f} psnr={rows[-1]['psnr']:.2f} "
                  f"t={rows[-1]['t_mean']:.3f} a={rows[-1]['a_mean']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "analysis", "prior_ablation.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pivot = df.pivot(index="t0", columns="a0", values="psnr")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pivot.plot(marker="o", ax=ax)
    ax.set_title("Init reconstruction PSNR vs physics-head prior biases")
    ax.set_xlabel("transmission prior t0")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(title="airlight a0")
    plt.tight_layout()
    fig.savefig(os.path.join(ROOT, "analysis", "figs", "prior_ablation.png"), dpi=150)
    plt.close(fig)

    print("\n=== best prior setting ===")
    print(df.sort_values("psnr", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
