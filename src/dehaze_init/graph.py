"""Computational-graph modeling of a deep dehazing network.

A network is treated as a directed acyclic graph (DAG) whose nodes are the
autograd operations recorded during one dummy forward pass, and whose edges are
tensor data-flow links (including main-path, residual/skip, dense-concat and
attention-gating branches).

For every conv / linear node we extract graph descriptors that an
initialization scheme can consume:

  * ``depth``       : longest path from an input source to this node
                      (critical-path depth, 0-based)
  * ``in_degree``   : number of incoming tensor branches
                      (1 = plain path, >=2 = concat / dense / residual add)
  * ``out_degree``  : number of downstream consumers
  * ``op_class``    : conv / linear / gate / concat / residual /
                      activation / pool / norm / other

The extraction runs on CPU and depends only on PyTorch's autograd graph, so it
works for any ``nn.Module`` without hard-coding architecture names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# op classification
# --------------------------------------------------------------------------- #
def _op_class(op_name: str) -> str:
    n = op_name.lower()
    if "conv" in n:
        return "conv"
    if "addmm" in n or "mm" in n or "linear" in n or "addmm" in n:
        return "linear"
    if "sigmoid" in n:
        return "gate"
    if "cat" in n:
        return "concat"
    if "add" in n:
        return "residual"
    if "relu" in n or "clamp" in n or "hardtanh" in n or "elu" in n:
        return "activation"
    if "pool" in n or "mean" in n or "adaptive" in n:
        return "pool"
    if "norm" in n:
        return "norm"
    return "other"


def _is_param_leaf(fn_name: str) -> bool:
    return "accumulategrad" in fn_name.lower()


@dataclass
class GraphInfo:
    """Container for the extracted DAG and per-module graph descriptors."""

    graph: nx.DiGraph
    module_nodes: Dict[str, Dict[str, object]] = field(default_factory=dict)
    summary: Dict[str, object] = field(default_factory=dict)

    def node_of(self, module_name: str) -> Dict[str, object]:
        return self.module_nodes.get(module_name, {})


def trace_graph(model: nn.Module, x: torch.Tensor) -> GraphInfo:
    """Run one forward pass and walk the autograd DAG.

    Parameters
    ----------
    model : nn.Module
    x : torch.Tensor
        A dummy input matching the model's expected shape.
    """
    model.eval()

    fn_to_name: Dict[int, str] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(name: str):
        def _h(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor) and t.grad_fn is not None:
                fn_to_name[id(t.grad_fn)] = name
        return _h

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(_hook(name)))

    with torch.enable_grad():
        out = model(x)
    for h in hooks:
        h.remove()

    G = nx.DiGraph()
    root = out.grad_fn
    stack: List[object] = [root]

    def _add(fn) -> Optional[int]:
        if fn is None:
            return None
        key = id(fn)
        if key not in G:
            op = type(fn).__name__
            G.add_node(
                key,
                op=op,
                op_class=_op_class(op),
                module=fn_to_name.get(key, ""),
            )
        return key

    while stack:
        fn = stack.pop()
        if fn is None:
            continue
        key = _add(fn)
        if key is None:
            continue
        op_name = type(fn).__name__
        if _is_param_leaf(op_name):
            continue  # parameter gradient leaf: not a data-flow node
        for next_fn, _idx in fn.next_functions:
            if next_fn is None:
                continue
            if _is_param_leaf(type(next_fn).__name__):
                continue
            nkey = _add(next_fn)
            if nkey is None:
                continue
            G.add_edge(nkey, key)  # data flows next_fn -> fn
            stack.append(next_fn)

    # topological longest-path depth
    depths: Dict[int, int] = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        depths[node] = 0 if not preds else 1 + max(depths[p] for p in preds)
    for node in G.nodes:
        G.nodes[node]["depth"] = depths[node]
        G.nodes[node]["in_degree"] = G.in_degree(node)
        G.nodes[node]["out_degree"] = G.out_degree(node)

    # per-module descriptor table
    module_nodes: Dict[str, Dict[str, object]] = {}
    for node, data in G.nodes(data=True):
        mod = data.get("module", "")
        if mod:
            # branch count: if a concat node feeds this node directly, inherit
            # the concat's in-degree (number of dense input branches).
            branch_in = int(data["in_degree"])
            for pred in G.predecessors(node):
                if G.nodes[pred]["op_class"] == "concat":
                    branch_in = max(branch_in, int(G.in_degree(pred)))
            module_nodes[mod] = {
                "depth": int(data["depth"]),
                "in_degree": int(data["in_degree"]),
                "branch_in": branch_in,
                "out_degree": int(data["out_degree"]),
                "op_class": data["op_class"],
                "op": data["op"],
            }

    # model-level summary
    class_counts: Dict[str, int] = {}
    for _n, data in G.nodes(data=True):
        cls = data["op_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    in_deg = [G.in_degree(n) for n in G.nodes]
    out_deg = [G.out_degree(n) for n in G.nodes]
    summary = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "critical_depth": max(depths.values()) if depths else 0,
        "density": nx.density(G),
        "avg_in_degree": sum(in_deg) / max(len(in_deg), 1),
        "avg_out_degree": sum(out_deg) / max(len(out_deg), 1),
        "n_conv_linear": class_counts.get("conv", 0) + class_counts.get("linear", 0),
        "n_residual_add": class_counts.get("residual", 0),
        "n_gate": class_counts.get("gate", 0),
        "n_concat": class_counts.get("concat", 0),
        "n_activation": class_counts.get("activation", 0),
        "n_pool": class_counts.get("pool", 0),
        "class_counts": ";".join(f"{k}:{v}" for k, v in sorted(class_counts.items())),
    }

    return GraphInfo(graph=G, module_nodes=module_nodes, summary=summary)


def graph_summary_rows(model_name: str, model: nn.Module,
                        x: torch.Tensor) -> Dict[str, object]:
    """Convenience: build the graph and return a flat summary row dict."""
    info = trace_graph(model, x)
    row: Dict[str, object] = {"model": model_name}
    row.update(info.summary)
    row["params"] = sum(p.numel() for p in model.parameters())
    return row


def node_table_rows(model_name: str, info: GraphInfo) -> List[Dict[str, object]]:
    """One row per traced conv/linear module."""
    rows: List[Dict[str, object]] = []
    for mod, desc in sorted(info.module_nodes.items()):
        rows.append({"model": model_name, "module": mod, **desc})
    return rows
