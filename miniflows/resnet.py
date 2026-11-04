# -*- coding: utf-8 -*-
"""
miniflows/resnet.py —— 基于 MiniFlows 的 ResNet（DonkeyCar 转向回归）
======================================================================

结构（与 CNN/ViT 版同一输入输出）：

    Stem: Conv(1→C0, k5, s2, p2) → GroupNorm → ReLU        (64→32)
    Stage1: 2 × BasicBlock(C0→C0, stride=1)                (32×32)
    Stage2: 2 × BasicBlock(C0→C1, stride=2) + 1×1 投影      (16×16)
    Stage3: 2 × BasicBlock(C1→C2, stride=2) + 1×1 投影      (8×8)
    GlobalAvgPool → Linear(C2→1) → Tanh

BasicBlock：

    y = ReLU( BN(Conv3x3(ReLU(BN(Conv3x3(x))))) + shortcut(x) )

为什么用 GroupNorm 而不是 BatchNorm：
    1. MiniFlows 目前没有 BatchNorm 的 running stats 持久化；
    2. GroupNorm 不依赖 batch 统计的滑动平均，训练/评估行为一致，
       save_model/load_model 只存参数也能逐位复现；
    3. 小 batch（32）下 GroupNorm 通常比 BN 更稳。
    代价是推理时要按组算均值/方差，略慢，但 64×64 输入下可接受。

参数量（默认 C0=8, widths=(8,16,32), blocks=(2,2,2)）约 4.3 万，
与 CNN 44,426 / ViT 44,353 基本对齐，便于公平比较。

【依赖】numpy + miniflows.graph/ops/layers（无循环依赖）。
"""

import numpy as np

from miniflows.graph import Node, param
from miniflows.layers import Layer, Linear, Conv2D, ReLU, Sequential, Tanh
from miniflows import ops


# ======================================================================
# 一、GroupNorm2D
# ======================================================================
class GroupNorm2D(Layer):
    """分组归一化：把 C 个通道分成 G 组，每组内对 (C/G, H, W) 做归一化。

    公式：
        x:(N,C,H,W) → reshape (N,G,C/G,H,W)
        μ,σ² = 每组在 (C/G,H,W) 上的均值/方差
        y = γ·(x-μ)/√(σ²+ε) + β
        reshape 回 (N,C,H,W)

    与 BatchNorm 的区别：不依赖 batch 维，训练/评估完全一致，无需 running stats。
    """

    def __init__(self, num_channels, num_groups=8, eps=1e-5, name=None):
        """参数：通道数、组数（必须整除通道数）、eps、层名。"""
        super().__init__(name)
        if num_channels % num_groups != 0:
            raise ValueError(f'通道数 {num_channels} 必须能被组数 {num_groups} 整除')
        self.num_channels = int(num_channels)
        self.num_groups = int(num_groups)
        self.eps = float(eps)
        self.gamma = param(np.ones(num_channels), name='gamma')
        self.beta = param(np.zeros(num_channels), name='beta')

    def forward(self, x):
        """x:(N,C,H,W) -> 同形状。"""
        N, C, H, W = x.data.shape
        G = self.num_groups
        # 1) 分组：(N,C,H,W) -> (N,G,C/G,H,W)
        xg = ops.reshape(x, (N, G, C // G, H, W))
        # 2) 组内均值/方差（对 C/G、H、W 三个轴）
        mu = ops.mean(xg, axis=(2, 3, 4), keepdims=True)
        xc = ops.sub(xg, mu)
        var = ops.mean(ops.mul(xc, xc), axis=(2, 3, 4), keepdims=True)
        y = ops.div(xc, ops.sqrt(ops.add(var, self.eps)))
        # 3) 还原形状并做仿射变换；γ/β 形状 (C,) 需补成 (1,C,1,1)
        y = ops.reshape(y, (N, C, H, W))
        gamma = ops.reshape(self.gamma, (1, C, 1, 1))
        beta = ops.reshape(self.beta, (1, C, 1, 1))
        return ops.add(ops.mul(y, gamma), beta)

    def named_params(self):
        """返回 [('gamma', Node), ('beta', Node)]。"""
        return [('gamma', self.gamma), ('beta', self.beta)]

    def params(self):
        """全部可训练参数。"""
        return [self.gamma, self.beta]


def _make_gn(channels, num_groups=None):
    """构造一个组数能整除通道数的 GroupNorm2D。"""
    g = num_groups if num_groups is not None else min(8, channels)
    while channels % g != 0:
        g -= 1
    return GroupNorm2D(channels, num_groups=g)


# ======================================================================
# 二、BasicBlock（ResNet-18/34 风格）
# ======================================================================
class BasicBlock(Layer):
    """两个 3×3 卷积 + 残差连接。

        y = ReLU( GN(Conv3x3(ReLU(GN(Conv3x3(x))))) + shortcut(x) )

    shortcut：
        - 形状不变：恒等映射；
        - stride≠1 或通道数变化：1×1 Conv + GN 投影。
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 num_groups=None, name=None):
        """参数：输入/输出通道、步幅、GroupNorm 组数、层名。"""
        super().__init__(name)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride=stride,
                            padding=1, name='conv1')
        self.gn1 = _make_gn(out_channels, num_groups)
        self.conv2 = Conv2D(out_channels, out_channels, 3, stride=1,
                            padding=1, name='conv2')
        self.gn2 = _make_gn(out_channels, num_groups)
        # 1×1 投影：通道或分辨率变化时必须用，否则残差无法相加
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2D(in_channels, out_channels, 1, stride=stride, padding=0,
                       name='proj_conv'),
                _make_gn(out_channels, num_groups),
            )
            self.shortcut.name = 'shortcut'
        else:
            self.shortcut = None

    def forward(self, x):
        """x:(N,C,H,W) -> (N,C',H',W')。"""
        identity = x if self.shortcut is None else self.shortcut(x)
        y = ops.relu(self.gn1(self.conv1(x)))     # 第一层 conv+GN+ReLU
        y = self.gn2(self.conv2(y))               # 第二层 conv+GN（先不加激活）
        return ops.relu(ops.add(y, identity))     # 残差相加后再 ReLU

    def named_params(self):
        """conv1/gn1/conv2/gn2/shortcut 的扁平参数名。"""
        out = []
        out.extend(_prefixed(self.conv1, 'conv1'))
        out.extend(_prefixed(self.gn1, 'gn1'))
        out.extend(_prefixed(self.conv2, 'conv2'))
        out.extend(_prefixed(self.gn2, 'gn2'))
        if self.shortcut is not None:
            out.extend(_prefixed(self.shortcut, 'shortcut'))
        return out

    def params(self):
        """全部可训练参数。"""
        return [p for _, p in self.named_params()]

    def train_mode(self, flag):
        """递归切换训练/评估模式（GroupNorm 无状态，但保留接口一致性）。"""
        self.conv1.train_mode(flag)
        self.gn1.train_mode(flag)
        self.conv2.train_mode(flag)
        self.gn2.train_mode(flag)
        if self.shortcut is not None:
            self.shortcut.train_mode(flag)


def _prefixed(module, prefix):
    """给子模块参数名加前缀。"""
    return [(f'{prefix}.{name}', p) for name, p in module.named_params()]


# ======================================================================
# 三、ResNet 回归器
# ======================================================================
class ResNetRegressor(Layer):
    """用于 DonkeyCar 转向角回归的小型 ResNet。

    Args:
        in_channels: 输入通道（灰度图=1）；
        c0: stem 输出通道；
        widths: 三个 stage 的输出通道；
        blocks: 每个 stage 的 BasicBlock 数；
        num_groups: GroupNorm 组数（默认取不超过 8 的最大可行值）。
    """

    def __init__(self, in_channels=1, c0=8, widths=(8, 16, 32),
                 blocks=(2, 2, 2), num_groups=None, name='resnet'):
        super().__init__(name)
        self.c0 = int(c0)
        self.widths = tuple(int(w) for w in widths)
        self.blocks_per_stage = tuple(int(b) for b in blocks)

        # Stem：64×64 → 32×32
        self.stem = Sequential(
            Conv2D(in_channels, c0, 5, stride=2, padding=2, name='stem_conv'),
            _make_gn(c0, num_groups),
            ReLU(),
        )
        self.stem.name = 'stem'

        # 三个 stage：32→32、16×16、8×8
        self.stages = []
        in_ch = c0
        for si, (out_ch, nb) in enumerate(zip(self.widths, self.blocks_per_stage)):
            stage = []
            for bi in range(nb):
                stride = 2 if (bi == 0 and si > 0) else 1
                stage.append(BasicBlock(in_ch, out_ch, stride=stride,
                                        num_groups=num_groups,
                                        name=f'block{bi}'))
                in_ch = out_ch
            container = Sequential(*stage)
            container.name = f'stage{si}'
            self.stages.append(container)

        self.head = Linear(self.widths[-1], 1, name='head')
        self.act = Tanh()

    def forward(self, x):
        """x:(N,1,H,W) -> (N,1) 转向角 ∈[-1,1]。"""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = ops.mean(x, axis=(2, 3))          # 全局平均池化 (N,C,1,1) -> (N,C)
        return self.act(self.head(x))          # (N,1)

    def named_params(self):
        """扁平化参数名，前缀与层结构一致。"""
        out = []
        out.extend(_prefixed(self.stem, 'stem'))
        for i, stage in enumerate(self.stages):
            out.extend(_prefixed(stage, f'stage{i}'))
        out.extend(_prefixed(self.head, 'head'))
        return out

    def params(self):
        """全部可训练参数。"""
        return [p for _, p in self.named_params()]

    def train_mode(self, flag):
        """递归切换训练/评估模式。"""
        self.stem.train_mode(flag)
        for stage in self.stages:
            stage.train_mode(flag)
        self.head.train_mode(flag)


def build_resnet(in_channels=1, c0=8, widths=(8, 16, 32), blocks=(2, 2, 2),
                 num_groups=None):
    """构造 DonkeyCar 用的 ResNet（默认约 4.3 万参数）。"""
    return ResNetRegressor(in_channels=in_channels, c0=c0, widths=widths,
                           blocks=blocks, num_groups=num_groups)
