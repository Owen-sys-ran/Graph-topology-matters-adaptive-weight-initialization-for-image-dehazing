"""Short real-training convergence comparison on synthetic ASM pairs.

This is a CPU-scale training study (not full RESIDE training): each
model x initialization scheme is trained for a fixed small number of steps
on a fixed pool of synthetic hazy/clean pairs, and we log the training-loss
curve, validation PSNR at init / during / at the end, and final gradient
health. It measures whether the initialization gives a healthy start and
early convergence on the same footing.

Outputs
-------
analysis/training_runs.csv        : one row per model x scheme x seed
analysis/training_curves.csv      : per-epoch train loss / val PSNR
analysis/figs/training_convergence.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dehaze_init import schemes, models, synthetic, probe  # noqa: E402

MODELS = ["aod", "dehazenet", "gfn", "ffa", "grid", "msbdn", "former"]
SCHEMES = ["default", "xavier", "kaiming", "dcgan", "depth_trunc", "dpi", "dpi_eps",
           "gai", "gai_eps"]
SEEDS = [0, 1]

SIZE = 32
TRAIN_N = 48
VAL_N = 32
BATCH = 8
EPOCHS = 12
LR = 1e-3
DEVICE = "cpu"


def apply_scheme(model, scheme: str, depth: int, seed: int):
    torch.manual_seed(seed)
    if scheme == "default":
        return
    if scheme == "dpi":
        schemes.dpi_init(model, depth=depth, seed=seed, residual_zero=True)
    elif scheme == "dpi_eps":
        schemes.dpi_init(model, depth=depth, seed=seed, residual_zero=False)
    elif scheme == "depth_trunc":
        schemes.depth_scaled_trunc(model, depth=depth)
    elif scheme == "gai":
        schemes.gai_init(model, depth=depth, seed=seed, residual_zero=True)
    elif scheme == "gai_eps":
        schemes.gai_init(model, depth=depth, seed=seed, residual_zero=False)
    else:
        schemes.SCHEMES[scheme](model)


@torch.no_grad()
def val_psnr(model, vi, vj, batch=16):
    model.eval()
    mse = 0.0
    n = 0
    for k in range(0, len(vi), batch):
        out = model(vi[k:k + batch]).clamp(0, 1)
        mse += torch.mean((out - vj[k:k + batch]) ** 2).item() * len(vi[k:k + batch])
        n += len(vi[k:k + batch])
    mse /= n
    return 99.0 if mse < 1e-8 else float(10 * np.log10(1.0 / mse))


def run(model_name, scheme, seed, ti, tj, vi, vj):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = models.build_model(model_name)
    apply_scheme(net, scheme, models.MODEL_DEPTH[model_name], seed)
    net.to(DEVICE)

    init = probe.measure(net, vi[: BATCH], vj[: BATCH], track_grad=True)
    init_psnr = val_psnr(net, vi, vj)

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = torch.nn.functional.mse_loss

    curves = []
    best_psnr = init_psnr
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(ti), generator=torch.Generator().manual_seed(seed * 100 + epoch))
        epoch_loss = 0.0
        nb = 0
        for k in range(0, len(ti), BATCH):
            idx = perm[k:k + BATCH]
            opt.zero_grad()
            out = net(ti[idx])
            loss = loss_fn(out, tj[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            nb += 1
        vp = val_psnr(net, vi, vj)
        best_psnr = max(best_psnr, vp)
        curves.append({"model": model_name, "scheme": scheme, "seed": seed,
                       "epoch": epoch, "train_loss": epoch_loss / nb, "val_psnr": vp})

    # final gradient health
    net.train()
    net.zero_grad(set_to_none=True)
    out = net(ti[:BATCH])
    loss_fn(out, tj[:BATCH]).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in net.parameters() if p.grad is not None], float("inf"))
    zero = sum(int((p.grad.abs() < 1e-8).sum()) for p in net.parameters() if p.grad is not None)
    total = sum(p.numel() for p in net.parameters() if p.grad is not None)

    final_loss = curves[-1]["train_loss"]
    final_psnr = curves[-1]["val_psnr"]
    row = {
        "model": model_name, "scheme": scheme, "seed": seed,
        "init_psnr": init_psnr, "init_residual": init["identity_residual"],
        "init_grad_norm": init["grad_norm"], "init_zero_grad": init["grad_zero_frac"],
        "final_train_loss": final_loss, "final_val_psnr": final_psnr,
        "best_val_psnr": best_psnr, "grad_norm_final": float(grad_norm),
        "zero_grad_final": zero / max(total, 1),
    }
    return row, curves


def main():
    os.makedirs(os.path.join(ROOT, "analysis", "figs"), exist_ok=True)
    ti, tj = synthetic.make_pool(TRAIN_N, SIZE, seed=100, device=DEVICE)
    vi, vj = synthetic.make_pool(VAL_N, SIZE, seed=200, device=DEVICE)

    rows, curves = [], []
    for m in MODELS:
        for scheme in SCHEMES:
            for seed in SEEDS:
                row, curve = run(m, scheme, seed, ti, tj, vi, vj)
                rows.append(row)
                curves.extend(curve)
                print(f"{m:10s} {scheme:12s} s{seed} initPSNR={row['init_psnr']:.2f} "
                      f"finalPSNR={row['final_val_psnr']:.2f} best={row['best_val_psnr']:.2f} "
                      f"loss={row['final_train_loss']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "analysis", "training_runs.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(ROOT, "analysis", "training_curves.csv"), index=False)

    # aggregate over seeds
    agg = df.groupby(["model", "scheme"]).agg(
        init_psnr=("init_psnr", "mean"),
        final_val_psnr=("final_val_psnr", "mean"),
        best_val_psnr=("best_val_psnr", "mean"),
        final_train_loss=("final_train_loss", "mean"),
        zero_grad_final=("zero_grad_final", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(ROOT, "analysis", "training_summary.csv"), index=False)

    _plot(df, pd.DataFrame(curves))
    print("\n=== training summary (mean over seeds) ===")
    print(agg.round(3).to_string(index=False))


def _plot(df, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = os.path.join(ROOT, "analysis", "figs")

    # convergence curves per model
    n = len(MODELS)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(13, 8), sharey=True)
    axes = axes.flatten()
    for i, m in enumerate(MODELS):
        ax = axes[i]
        sub = curves[curves["model"] == m]
        for scheme in SCHEMES:
            c = sub[sub["scheme"] == scheme].groupby("epoch")["val_psnr"].mean()
            ax.plot(c.index, c.values, marker="o", ms=3, label=scheme)
        ax.set_title(m)
        ax.set_xlabel("epoch")
        if i % 3 == 0:
            ax.set_ylabel("val PSNR (dB)")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "training_convergence.png"), dpi=150)
    plt.close(fig)

    # final PSNR grouped bar
    fig, ax = plt.subplots(figsize=(13, 5))
    pivot = df.groupby(["model", "scheme"])["final_val_psnr"].mean().unstack()
    pivot = pivot.reindex(MODELS)[SCHEMES]
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Final validation PSNR after short synthetic training")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=7, ncol=4)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "training_final_psnr.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
