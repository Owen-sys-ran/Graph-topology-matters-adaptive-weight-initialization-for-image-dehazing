"""Analyze pretrained dehazing checkpoints available locally.

Outputs
-------
analysis/checkpoint_weight_stats.csv  : per-tensor statistics
analysis/checkpoint_summary.csv       : per-checkpoint / per-role summary
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dehaze_init.weight_stats import analyze_state_dict, strip_ddp  # noqa: E402

REPOS = os.path.join(ROOT, "reference_repos")

CHECKPOINTS = [
    ("GridDehazeNet", "indoor", os.path.join(REPOS, "GridDehazeNet", "indoor_haze_best_3_6")),
    ("GridDehazeNet", "outdoor", os.path.join(REPOS, "GridDehazeNet", "outdoor_haze_best_3_6")),
    ("AOD-Net(repro)", "ots", os.path.join(REPOS, "PyTorch-Image-Dehazing", "snapshots", "dehazer.pth")),
    ("ED-Former", "indoor", os.path.join(REPOS, "ED-Former", "saved_models", "indoor", "ed-former.pth")),
    ("ED-Former", "outdoor", os.path.join(REPOS, "ED-Former", "saved_models", "outdoor", "ed-former.pth")),
    ("ED-Former", "rshaze", os.path.join(REPOS, "ED-Former", "saved_models", "rshaze", "ed-former.pth")),
    ("SLRNet", "RESIDE-6K", os.path.join(REPOS, "SLRNet", "weight.pth")),
]


def _load_sd(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    return strip_ddp(obj)


def main() -> None:
    os.makedirs(os.path.join(ROOT, "analysis"), exist_ok=True)
    all_rows = []
    summaries = []

    for family, split, path in CHECKPOINTS:
        if not os.path.exists(path):
            print(f"[skip] {family}/{split}: {path} not found")
            continue
        try:
            sd = _load_sd(path)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {family}/{split}: {repr(e)[:120]}")
            continue

        rows = analyze_state_dict(sd)
        n_params = sum(r["numel"] for r in rows)
        print(f"[ok] {family}/{split}: tensors={len(rows)} params={n_params}")

        for r in rows:
            r["family"] = family
            r["split"] = split
            all_rows.append(r)

        conv_rows = [r for r in rows if r["role"] in ("conv", "head", "gate", "qk")]
        summaries.append({
            "family": family,
            "split": split,
            "tensors": len(rows),
            "params": n_params,
            "weight_std_mean": pd.Series([r["std"] for r in conv_rows]).mean(),
            "weight_std_median": pd.Series([r["std"] for r in conv_rows]).median(),
            "abs_weight_mean": pd.Series([r["abs_mean"] for r in conv_rows]).mean(),
            "sparsity": pd.Series([r["sparsity"] for r in conv_rows]).mean(),
            "spectral_norm_mean": pd.Series([r["spectral_norm"] for r in conv_rows
                                             if r["spectral_norm"] == r["spectral_norm"]]).mean(),
        })

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(ROOT, "analysis", "checkpoint_weight_stats.csv"), index=False)
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(os.path.join(ROOT, "analysis", "checkpoint_summary.csv"), index=False)
    print("\n=== checkpoint summary ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
