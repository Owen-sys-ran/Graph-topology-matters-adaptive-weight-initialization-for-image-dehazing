# GAI-Dehaze: Graph-Aware Initialization for Image Dehazing

**Topology-driven weight initialization for deep image dehazing networks.**

> Weight initialization shapes the optimization trajectory of a dehazing model: whether
> signals vanish or amplify across layers, whether gradients reach early layers, and
> whether the first output stays in a reasonable image range. Conventional schemes
> (Xavier, He/Kaiming, DCGAN-style normal, truncated normal) assign a uniform variance
> rule to every layer and ignore how the network is actually connected.
>
> This repository treats each dehazing network as a **directed acyclic computation
> graph**, extracts per-node topological descriptors, and proposes two
> dehazing-specific initializations:
>
> - **DPI** — Dehazing-aware Parameter Initialization: depth-scaled variance keeping
>   rules, role-specific head priors, near-identity residual start.
> - **GAI** — Graph-Aware Initialization (proposed): per-node variance budget allocated
>   from critical-path depth and dense-branch count, generalizing DPI to graph topology.

## Abstract

Weight initialization critically governs convergence stability and final restoration
quality of deep dehazing models. Existing initialization schemes disregard the
intrinsic computation graph topology of dehazing networks — residual skip connections,
multi-branch concatenation, dense feature fusion and attention pathways — which often
induce forward signal amplification and gradient instability, especially in deep
transformer-based dehazing architectures.

We model dehazing networks as directed acyclic graphs (DAGs), automatically traced from
the autograd graph of one dummy forward pass, and extract per-node descriptors:
critical-path depth, dense-branch fan-in, residual/add nodes, attention gates and
concat nodes. Based on forward/backward variance propagation theory under the
atmospheric scattering model prior, we propose **Graph-Aware Initialization (GAI)**:
each conv/linear node receives an adaptive variance budget conditioned on its graph
position, instead of a single global standard deviation.

We evaluate seven lightweight reproduced dehazing backbones against nine initialization
schemes: pre-trained checkpoint weight statistics, initial forward/gradient probing,
and a fixed-pool 12-epoch, two-seed short-training convergence comparison on synthetic
hazy-clean pairs. On deep residual/transformer dehazing networks, GAI and DPI converge
stably and rank among the best; GAI-Eps achieves the best final validation PSNR on
FFA-Net, DehazeFormer and GridDehazeNet. He/Kaiming diverges on the transformer stack.
On shallow formula-reconstruction networks, classical Xavier/He remain competitive,
showing that initialization payoff depends on depth and graph topology.

## Method

### 1. Computation graph

A network is treated as a DAG `G = (V, E)`: every autograd operation recorded during a
dummy forward pass is a node; tensor data-flow links are edges. Nodes are classified as
conv/linear, gate (sigmoid), residual add, dense concat, activation, pooling or norm.
For every learnable node `v` we record:

| Descriptor | Meaning |
|---|---|
| `d_v` | longest-path (critical-path) depth from an input source |
| `ρ_v = d_v / D_crit` | normalized path position in `[0, 1]` |
| `k_v` | dense branch count; inherited from a feeding concat node's in-degree |
| role | body / QK projection / attention gate / reconstruction head / physics head / norm |

### 2. GAI variance budget

Starting from the He bandwidth `σ0 = sqrt(2 / fan_in)`:

- depth attenuation `g_v = [ 8 · (1 + ρ_v (D − 1)) ]^(−1/4)`: stem nodes keep the
  largest budget, nodes on the residual end of the critical path receive the smallest;
- branch compensation `b_v = 1 / sqrt(k_v)`: `k` fused branches do not sum their
  second moments uncontrolled.

```
σ_v = g_v · b_v · sqrt(2 / fan_in)     # body layers
```

Role rules are shared with DPI:

- Q/K projections: Xavier truncated normal, `σ = sqrt(2/(fan_in+fan_out))`;
- attention gate last conv: small He scale, bias 0 (gate starts at 0.5);
- reconstruction head: zero-init (**GAI-Identity**, network starts as the hazy input)
  or 0.1× small scale (**GAI-Eps**, preserves near-identity start while keeping first-step
  gradient flow to the body);
- transmission / atmospheric-light heads: prior-biased sigmoid bias `logit(t0)`,
  `logit(a0)`;
- normalization: weight 1, bias 0.

When all nodes share a similar path depth and there are no multi-branch concats, GAI
degenerates to the global depth-gain DPI.

## Repository structure

```
gai-dehaze/
├── README.md
├── LICENSE
├── CITATION.bib
├── requirements.txt
├── docs/
│   └── REPORT_EN.md                # full English technical report with tables & figures
├── src/
│   ├── dehaze_init/
│   │   ├── schemes.py              # baseline + DPI + GAI schemes, forward variance calibration
│   │   ├── models.py               # seven lightweight reproduced backbones
│   │   ├── synthetic.py            # hazy-clean pairs from the atmospheric scattering model
│   │   ├── weight_stats.py          # per-tensor checkpoint weight statistics
│   │   ├── graph.py                # autograd DAG tracer and topology descriptors
│   │   └── probe.py                 # initial forward / gradient probe
│   └── experiments/
│       ├── run_checkpoint_analysis.py
│       ├── run_graph_analysis.py
│       ├── run_init_comparison.py
│       ├── run_training_experiment.py
│       ├── run_prior_ablation.py
│       └── run_weight_plots.py
├── analysis/                       # CSV results and generated figures
│   ├── graph_summary.csv
│   ├── graph_nodes.csv
│   ├── init_comparison_summary.csv
│   ├── training_summary.csv
│   ├── prior_ablation.csv
│   └── figs/
└── reference_repos/                # cloned public repos used only for weight statistics
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
cd src/experiments

# 1. real checkpoint weight statistics
python run_checkpoint_analysis.py

# 2. computation-graph topology analysis (DAG metrics + GAI budget figures)
python run_graph_analysis.py

# 3. initial forward / gradient probe, nine schemes × seven models, three seeds
python run_init_comparison.py

# 4. fixed-pool short-training convergence comparison, two seeds, 12 epochs
python run_training_experiment.py

# 5. transmission / atmospheric-light prior bias ablation
python run_prior_ablation.py

# 6. weight statistics figures
python run_weight_plots.py
```

All CSV outputs go to `analysis/`; figures to `analysis/figs/`.

## Evaluated backbones

| Key | Model | Architecture note |
|---|---|---|
| `aod` | AOD-Net | five-conv cross-layer concat, reformulated scattering model |
| `dehazenet` | DehazeNet | Maxout + BReLU, explicit t/A heads and model inversion |
| `gfn` | GFN | white-balance / contrast / gamma branches + softmax confidence fusion |
| `ffa` | FFA-Net | channel & pixel attention residual blocks |
| `grid` | GridDehazeNet | residual dense blocks + channel attention |
| `msbdn` | MSBDN | multi-level dense fusion |
| `former` | DehazeFormer | depthwise conv + 1×1 proj + MLP residual stack |

## Initialization schemes compared

`default` (PyTorch default), `xavier`, `kaiming` (He), `dcgan` (N(0, 0.02)),
`depth_trunc` (depth-scaled truncated normal), `dpi` (DPI-Identity), `dpi_eps`
(DPI-Eps), `gai` (GAI-Identity), `gai_eps` (GAI-Eps).

## Key results

Short-training final validation PSNR (dB, two-seed average; full table in
[`docs/REPORT_EN.md`](docs/REPORT_EN.md)):

| Model | Xavier | Kaiming | DCGAN | DPI | DPI-Eps | GAI | GAI-Eps |
|---|---|---|---|---|---|---|---|
| AOD-Net | 17.19 | 16.90 | 17.23 | 17.38 | 17.38 | 16.66 | 16.66 |
| DehazeNet | 17.91 | 17.81 | 16.06 | 16.60 | 16.60 | 17.42 | 17.42 |
| FFA-Net | 20.57 | 14.73 | 21.35 | 21.76 | 21.75 | 21.79 | **21.80** |
| DehazeFormer | 18.95 | 5.62 | 21.36 | 21.87 | 21.86 | 21.89 | **21.91** |
| GFN | 14.55 | 13.13 | 15.08 | **16.05** | 16.05 | 15.81 | 15.81 |
| GridDehazeNet | 19.93 | 15.18 | 21.23 | 21.51 | 21.62 | 21.57 | **21.68** |
| MSBDN | 21.06 | 18.52 | 21.15 | 21.23 | 21.23 | 21.19 | 21.19 |

- Deep residual / transformer stacks: depth-scaled schemes, DCGAN, DPI and GAI are
  stable; Kaiming/ Xavier amplify signals and Kaiming diverges on the transformer.
- Shallow multi-branch nets: classical Xavier/He stay competitive; graph-aware depth
  scaling yields no extra gain there.

## Scope and limitations

Experiments run on CPU with lightweight reproduced models and 32×32 synthetic pairs;
the 12-epoch short training captures early-convergence trends, not final accuracy.
Full-scale validation on RESIDE-ITS/OTS and other real dehazing datasets, multiple
seeds, and final PSNR/SSIM remain GPU future work. See `docs/REPORT_EN.md` §9.2.

## Reference repositories

`reference_repos/` contains public implementations cloned for checkpoint statistics
and initialization-source inspection only. Copyright and license of each repo remain
with its original authors: AOD-Net, DCMPNet, DehazeFormer, ED-Former, FFA-Net,
GridDehazeNet, PyTorch-Image-Dehazing, SLRNet.

## Citation

```bibtex
@misc{gai_dehaze2026,
  title  = {Graph-Aware Initialization for Image Dehazing},
  author = {GAI-Dehaze Authors},
  year   = {2026},
  howpublished = {\url{https://github.com/your-org/gai-dehaze}}
}
```

## License

Code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)).
Third-party reference repos keep their original licenses.
