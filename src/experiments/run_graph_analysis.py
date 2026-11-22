"""Computational-graph topology analysis of the seven reference dehazing nets.

For every model we trace the autograd DAG on a dummy forward pass, record
graph descriptors (nodes, edges, critical-path depth, degrees, op-class counts),
then run GAI and record the empirical per-node weight std against its graph
depth, validating the depth / branch variance budget.

Outputs
-------
analysis/graph_summary.csv   : one row per model
analysis/graph_nodes.csv    : one row per traced conv/linear module
analysis/figs/graph_topology.png
analysis/figs/graph_dag_<model>.png
analysis/figs/graph_depth_std.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dehaze_init import models, schemes, graph  # noqa: E402

MODELS = ["aod", "dehazenet", "gfn", "ffa", "grid", "msbdn", "former"]
SIZE = 32


def _weight_std(module: nn.Module) -> dict:
    out = {}
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            out[name] = float(m.weight.detach().std(unbiased=False))
    return out


def main() -> None:
    os.makedirs(os.path.join(ROOT, "analysis", "figs"), exist_ok=True)
    x = torch.rand(1, 3, SIZE, SIZE)

    summary_rows, node_rows = [], []
    traced = {}
    for mname in MODELS:
        net = models.build_model(mname)
        info = graph.trace_graph(net, x)
        traced[mname] = (net, info)
        row = graph.graph_summary_rows(mname, net, x)
        summary_rows.append(row)

        # GAI allocation, then empirical per-node std
        schemes.gai_init(net, depth=models.MODEL_DEPTH[mname], seed=0)
        wstd = _weight_std(net)
        for nr in graph.node_table_rows(mname, info):
            nr["weight_std_gai"] = wstd.get(nr["module"], np.nan)
            node_rows.append(nr)
        print(f"{mname:10s} nodes={row['n_nodes']:3d} edges={row['n_edges']:3d} "
              f"depth={row['critical_depth']:3d} adds={row['n_residual_add']} "
              f"gates={row['n_gate']} cats={row['n_concat']}")

    sd = pd.DataFrame(summary_rows)
    nd = pd.DataFrame(node_rows)
    sd.to_csv(os.path.join(ROOT, "analysis", "graph_summary.csv"), index=False)
    nd.to_csv(os.path.join(ROOT, "analysis", "graph_nodes.csv"), index=False)

    _plot(sd, nd, traced)
    print("\n=== graph summary ===")
    print(sd[["model", "n_nodes", "n_edges", "critical_depth", "n_residual_add",
              "n_gate", "n_concat", "density"]].round(3).to_string(index=False))


def _plot(sd: pd.DataFrame, nd: pd.DataFrame, traced: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    figs = os.path.join(ROOT, "analysis", "figs")
    models_order = MODELS

    # 1) topology bars: nodes / edges / critical depth
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, title in zip(
        axes,
        ["n_nodes", "n_edges", "critical_depth"],
        ["operation nodes", "data-flow edges", "critical-path depth"],
    ):
        vals = sd.set_index("model").reindex(models_order)[col]
        ax.bar(vals.index, vals.values, color="#4C78A8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("count")
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "graph_topology.png"), dpi=150)
    plt.close(fig)

    # 2) DAG drawings for two representative architectures
    op_color = {
        "conv": "#4C78A8", "linear": "#72B7B2", "gate": "#F58518",
        "concat": "#E45756", "residual": "#54A24B", "activation": "#B279A2",
        "pool": "#9D755D", "norm": "#FF9DA6", "other": "#9D9D9D",
    }
    for mname in ["ffa", "former"]:
        _net, info = traced[mname]
        G = info.graph
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=7, k=0.9)
        colors = [op_color.get(G.nodes[n]["op_class"], "#999") for n in G.nodes]
        sizes = [60 + 40 * G.out_degree(n) for n in G.nodes]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, arrows=True,
                               arrowsize=6, node_size=sizes)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes)
        labels = {n: G.nodes[n]["module"]
                  for n in G.nodes if G.nodes[n]["module"]}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)
        ax.set_title(f"Computational DAG — {mname}")
        ax.axis("off")
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=k)
                   for k, c in op_color.items() if k in
                   {G.nodes[n]["op_class"] for n in G.nodes}]
        ax.legend(handles=handles, fontsize=7, loc="upper left", ncol=2)
        plt.tight_layout()
        fig.savefig(os.path.join(figs, f"graph_dag_{mname}.png"), dpi=150)
        plt.close(fig)

    # 3) empirical GAI weight std vs critical-path depth
    fig, ax = plt.subplots(figsize=(8, 5))
    for mname in models_order:
        sub = nd[nd["model"] == mname]
        ax.scatter(sub["depth"], sub["weight_std_gai"], s=35, label=mname, alpha=0.8)
    ax.set_xlabel("node critical-path depth")
    ax.set_ylabel("GAI weight std")
    ax.set_title("Per-node weight std vs graph depth under GAI")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "graph_depth_std.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
