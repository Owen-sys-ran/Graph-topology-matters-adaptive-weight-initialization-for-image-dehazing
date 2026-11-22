"""Compare initialization schemes at init time on tiny dehazing models.

For every model x scheme x seed we build a fresh network, apply the
initialization, and probe forward signal flow + backward gradient flow on a
synthetic ASM hazy pair. No optimizer step / training is performed.

Outputs
-------
analysis/init_comparison_long.csv     : raw per-run metrics
analysis/init_comparison_summary.csv  : mean +/- std over seeds
analysis/figs/*.png                   : comparison figures
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
SEEDS = [0, 1, 2]
BATCH, SIZE = 4, 64


def apply_scheme(model: torch.nn.Module, scheme: str, depth: int, seed: int):
    torch.manual_seed(seed)
    if scheme == "default":
        return  # keep PyTorch native default
    if scheme == "dpi":
        schemes.dpi_init(model, depth=depth, seed=seed, residual_zero=True)
    elif scheme == "dpi_eps":
        # Identity-oriented but non-zero head: small-scale reconstruction conv,
        # keeps first-step gradient flowing into the body.
        schemes.dpi_init(model, depth=depth, seed=seed, residual_zero=False)
    elif scheme == "depth_trunc":
        schemes.depth_scaled_trunc(model, depth=depth)
    elif scheme == "gai":
        schemes.gai_init(model, depth=depth, seed=seed, residual_zero=True)
    elif scheme == "gai_eps":
        schemes.gai_init(model, depth=depth, seed=seed, residual_zero=False)
    else:
        schemes.SCHEMES[scheme](model)


def build(name: str):
    return models.build_model(name)


def main() -> None:
    os.makedirs(os.path.join(ROOT, "analysis", "figs"), exist_ok=True)
    x, j, _, _ = synthetic.make_pair(BATCH, SIZE, seed=0)

    rows = []
    for mname in MODELS:
        depth = models.MODEL_DEPTH[mname]
        for scheme in SCHEMES:
            for seed in SEEDS:
                net = build(mname)
                apply_scheme(net, scheme, depth, seed)
                m = probe.measure(net, x, j, track_grad=True)
                m.update({"model": mname, "scheme": scheme, "seed": seed})
                rows.append(m)
                print(f"{mname:6s} {scheme:12s} seed={seed} "
                      f"id_res={m['identity_residual']:.3f} "
                      f"psnr={m['psnr_vs_clean']:.2f} "
                      f"grad_norm={m['grad_norm']:.3f} "
                      f"zero_grad={m['grad_zero_frac']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "analysis", "init_comparison_long.csv"), index=False)

    metrics = ["identity_residual", "psnr_vs_clean", "grad_norm", "grad_zero_frac",
               "loss_at_init", "act_std_min", "act_std_max", "grad_std_body"]
    agg = {}
    for met in metrics:
        g = df.groupby(["model", "scheme"])[met].agg(["mean", "std"]).reset_index()
        g.columns = ["model", "scheme", f"{met}_mean", f"{met}_std"]
        agg[met] = g
    summary = agg[metrics[0]]
    for met in metrics[1:]:
        summary = summary.merge(agg[met], on=["model", "scheme"])
    summary.to_csv(os.path.join(ROOT, "analysis", "init_comparison_summary.csv"), index=False)

    _plot(df, summary)
    print("\n=== summary (identity residual, lower is better; PSNR higher) ===")
    cols = ["model", "scheme", "identity_residual_mean", "psnr_vs_clean_mean",
            "grad_norm_mean", "grad_zero_frac_mean"]
    print(summary[cols].to_string(index=False))


def _plot(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir = os.path.join(ROOT, "analysis", "figs")
    schemes_order = SCHEMES

    # 1) identity residual & PSNR grouped bar
    for metric, title, ylabel, logy in [
        ("identity_residual", "Initial identity residual ||out-I||/||I|| (lower better)", "residual", True),
        ("psnr_vs_clean", "Initial PSNR vs clean (higher better)", "dB", False),
        ("grad_zero_frac", "Fraction of zero gradients (lower better)", "fraction", False),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot = summary.pivot(index="scheme", columns="model", values=f"{metric}_mean")
        pivot = pivot.reindex(schemes_order)
        pivot.plot(kind="bar", ax=ax)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.legend(title="model")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(figs_dir, f"{metric}.png"), dpi=150)
        plt.close(fig)

    # 2) activation std depth profile per model, one run per scheme
    for mname in MODELS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sub = df[df["model"] == mname]
        for scheme in schemes_order:
            run = sub[sub["scheme"] == scheme].iloc[0]
            # reconstruct layer profile from a fresh probe
            net = build(mname)
            apply_scheme(net, scheme, models.MODEL_DEPTH[mname], 0)
            x, j, _, _ = synthetic.make_pair(BATCH, SIZE, seed=0)
            cap = probe._forward_capture(net, x)
            ax.plot(cap["layer_act_std"], marker="o", label=scheme)
        ax.set_title(f"Activation std by layer — {mname}")
        ax.set_ylabel("activation std")
        ax.set_xlabel("layer index")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(figs_dir, f"act_profile_{mname}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
