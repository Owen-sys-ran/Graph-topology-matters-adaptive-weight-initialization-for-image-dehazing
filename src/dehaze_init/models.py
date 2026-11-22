"""Tiny CPU-friendly reference dehazing architectures.

These are faithful-but-compact re-implementations of three representative
families, used to compare initialization schemes at init time (no training):

  * ``TinyAOD``    -- reformulated ASM network, mirrors the ICCV'17 AOD-Net
                       5-concatenated-conv layout.
  * ``TinyFFA``     -- residual attention network with channel / pixel attention,
                       mirrors AAAI'20 FFA-Net block/group structure.
  * ``TinyFormer``  -- residual transformer-style block stack with depthwise
                       conv, 1x1 projections and an MLP, mirrors the
                       TIP'23 DehazeFormer block.

All forward passes run on CPU at tiny resolution (e.g. 64x64).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# AOD-Net family
# --------------------------------------------------------------------------- #
class TinyAOD(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(cat3))
        clean = self.relu((x5 * x) - x5 + 1.0)
        return clean


# --------------------------------------------------------------------------- #
# FFA-Net family
# --------------------------------------------------------------------------- #
class _PALayer(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, max(channel // 8, 1), 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channel // 8, 1), 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.pa(x)


class _CALayer(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, max(channel // 8, 1), 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channel // 8, 1), channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.ca(self.avg_pool(x))


class _FFABlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.calayer = _CALayer(dim)
        self.palayer = _PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        return res + x


class TinyFFA(nn.Module):
    def __init__(self, dim: int = 16, blocks: int = 2):
        super().__init__()
        self.pre = nn.Conv2d(3, dim, 3, padding=1, bias=True)
        self.blocks = nn.Sequential(*[_FFABlock(dim) for _ in range(blocks)])
        self.post = nn.Conv2d(dim, 3, 3, padding=1, bias=True)

    def forward(self, x):
        f = self.pre(x)
        f = self.blocks(f)
        # Global residual: network predicts the restoration residual.
        return x + self.post(f)


# --------------------------------------------------------------------------- #
# DehazeFormer family (compact)
# --------------------------------------------------------------------------- #
class _FormerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=True)
        self.V = nn.Conv2d(dim, dim, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 1, bias=True),
        )

    def forward(self, x):
        y = self.dw(x)
        y = self.proj(self.V(y))
        x = x + y
        x = x + self.mlp(x)
        return x


class TinyFormer(nn.Module):
    def __init__(self, dim: int = 16, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.stem = nn.Conv2d(3, dim, 3, padding=1, bias=True)
        self.blocks = nn.Sequential(*[_FormerBlock(dim) for _ in range(depth)])
        self.head_out = nn.Conv2d(dim, 3, 3, padding=1, bias=True)

    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        # Global residual: network predicts the restoration residual.
        return x + self.head_out(f)


# --------------------------------------------------------------------------- #
# DehazeNet family: explicit transmission + atmospheric light, ASM inversion
# --------------------------------------------------------------------------- #
class _BReLU(nn.Module):
    """Bilateral ReLU: clamp activation to [0, 1] (DehazeNet-style)."""

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class TinyDehazeNet(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.maxout = nn.Conv2d(3, dim * 2, 3, padding=1)
        self.brelu = _BReLU()
        self.body1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.body2 = nn.Conv2d(dim, dim, 3, padding=1)
        # explicit physics heads
        self.t_head = nn.Conv2d(dim, 1, 3, padding=1)
        self.a_head = nn.Conv2d(dim, 3, 1)

    def forward(self, x):
        f = self.maxout(x)
        f = f.reshape(f.shape[0], 2, -1, f.shape[2], f.shape[3]).max(dim=1).values
        f = self.brelu(f)
        f = self.brelu(self.body1(f))
        f = self.brelu(self.body2(f))
        t = torch.sigmoid(self.t_head(f)).clamp(1e-3, 1.0)
        a = torch.sigmoid(self.a_head(f)).mean(dim=(2, 3), keepdim=True)
        j = (x - a * (1.0 - t)) / t
        return j.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# GFN family: three preprocessed branches with learned confidence fusion
# --------------------------------------------------------------------------- #
class TinyGFN(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, 3, 3, padding=1),
        )
        self.conf = nn.Conv2d(3, 1, 1)

    @staticmethod
    def _preprocess(x: torch.Tensor) -> list:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-6
        wb = (x / (mean + 1e-6)) * 0.5
        contrast = (x - mean) / std * 0.5 + 0.5
        gamma = x.pow(2.2)
        return [wb.clamp(0, 1), contrast.clamp(0, 1), gamma.clamp(0, 1)]

    def forward(self, x):
        branches = self._preprocess(x)
        outs, confs = [], []
        for b in branches:
            outs.append(self.branch(b))
            confs.append(self.conf(b))
        conf = torch.softmax(torch.cat(confs, dim=1), dim=1)  # B,3,H,W
        j = sum(conf[:, k:k + 1] * outs[k] for k in range(3))
        return j.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# GridDehazeNet family: residual dense block + channel attention
# --------------------------------------------------------------------------- #
class _RDB(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.c2 = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.c3 = nn.Conv2d(dim * 3, dim, 3, padding=1)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        h1 = torch.relu(self.c1(x))
        h2 = torch.relu(self.c2(torch.cat([x, h1], 1)))
        h3 = torch.relu(self.c3(torch.cat([x, h1, h2], 1)))
        h3 = h3 * self.att(h3)
        return x + h3


class TinyGrid(nn.Module):
    def __init__(self, dim: int = 8, blocks: int = 2):
        super().__init__()
        self.stem = nn.Conv2d(3, dim, 3, padding=1)
        self.rdbs = nn.Sequential(*[_RDB(dim) for _ in range(blocks)])
        self.head_out = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x):
        f = self.stem(x)
        f = self.rdbs(f)
        return x + self.head_out(f)


# --------------------------------------------------------------------------- #
# MSBDN family: dense multi-level feature fusion
# --------------------------------------------------------------------------- #
class TinyMSBDN(nn.Module):
    def __init__(self, dim: int = 8, levels: int = 3):
        super().__init__()
        self.levels = levels
        self.convs = nn.ModuleList(
            [nn.Conv2d(3 if i == 0 else dim, dim, 3, padding=1) for i in range(levels)]
        )
        self.fuse = nn.Conv2d(dim * levels, dim, 1)
        self.head_out = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x):
        feats = []
        h = x
        for i, conv in enumerate(self.convs):
            h = torch.relu(conv(h if i == 0 else feats[-1]))
            feats.append(h)
        f = self.fuse(torch.cat(feats, dim=1))
        return x + self.head_out(f)


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("aod", "tinyaod"):
        return TinyAOD()
    if name in ("ffa", "tinyffa"):
        return TinyFFA(**kwargs)
    if name in ("former", "tinyformer"):
        return TinyFormer(**kwargs)
    if name in ("dehazenet", "dn"):
        return TinyDehazeNet(**kwargs)
    if name in ("gfn",):
        return TinyGFN(**kwargs)
    if name in ("grid", "gridrdb"):
        return TinyGrid(**kwargs)
    if name in ("msbdn",):
        return TinyMSBDN(**kwargs)
    raise ValueError(f"unknown model {name}")


MODEL_DEPTH = {"aod": 5, "ffa": 4, "former": 4, "dehazenet": 4,
               "gfn": 2, "grid": 4, "msbdn": 3}
