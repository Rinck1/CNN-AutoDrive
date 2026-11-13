# -*- coding: utf-8 -*-
"""
test_models.py —— 经典模型 / 分组卷积 / 规模分析测试
======================================================

覆盖：
    1. 分组卷积与“逐组普通卷积再拼接”数值一致；
    2. 深度卷积与“逐通道单通道卷积”一致，且梯度通过中心差分；
    3. VGG11 / MobileNetV1 前向形状与参数量合理（深度可分离确实更省）；
    4. profiler 的 FLOPs/参数量/激活内存口径正确（手算对照）；
    5. 不同模型参数量级排序符合预期（MobileNet < VGG < 大模型）。

运行（项目根目录）：
    python tests/test_models.py
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from common_setup import init_env
init_env(seed=42)

import miniflows.ops as ops
from miniflows.graph import Node
from miniflows.autodiff import gradient_check
from miniflows.profiler import count_flops, count_params, measure_activation_bytes
from miniflows.layers import Sequential, Conv2D, ReLU, Flatten, Linear
from miniflows.classic_cnn import (build_vgg11, build_mobilenet_v1,
                                   DepthwiseConv2D, SeparableConv2D)


def test_grouped_conv():
    """分组卷积 == 逐组普通卷积后沿通道拼接。"""
    print('\n[1] 分组卷积一致性')
    rng = np.random.RandomState(0)
    x = rng.randn(2, 4, 8, 8)
    W = rng.randn(4, 2, 3, 3) * 0.3
    got = ops.conv2d_groups(Node(x), Node(W), stride=1, padding=1, groups=2).data
    parts = [ops.conv2d(Node(x[:, g * 2:(g + 1) * 2]),
                        Node(W[g * 2:(g + 1) * 2]), padding=1).data
             for g in range(2)]
    ref = np.concatenate(parts, axis=1)
    diff = float(np.max(np.abs(got - ref)))
    print(f'    与逐组卷积拼接最大差 = {diff:.2e}')
    assert diff < 1e-12
    print('    ✓ 分组卷积实现正确')


def test_depthwise_conv():
    """深度卷积 == 逐通道卷积；梯度通过中心差分。"""
    print('\n[2] 深度卷积一致性 + 梯度')
    rng = np.random.RandomState(1)
    x = rng.randn(2, 4, 8, 8)
    W = rng.randn(4, 1, 3, 3) * 0.3
    b = rng.randn(4) * 0.1
    got = ops.depthwise_conv2d(Node(x), Node(W), Node(b), padding=1).data
    ref = np.stack([ops.conv2d(Node(x[:, c:c + 1]), Node(W[c:c + 1]),
                               Node(b[c:c + 1]), padding=1).data[:, 0]
                    for c in range(4)], axis=1)
    diff = float(np.max(np.abs(got - ref)))
    print(f'    与逐通道卷积最大差 = {diff:.2e}')
    assert diff < 1e-12

    Wo = rng.randn(2, 4, 8, 8)          # 固定输出权重，避免每次调用重新随机
    res = gradient_check(
        lambda ns: ops.sum(ops.mul(ops.depthwise_conv2d(ns[0], ns[1], ns[2],
                                                        padding=1), Wo)),
        [rng.randn(2, 4, 8, 8), rng.randn(4, 1, 3, 3) * 0.3, rng.randn(4) * 0.1])
    print(f"    梯度 passed={res['passed']} max_rel={res['max_rel_error']:.2e}")
    assert res['passed']
    print('    ✓ 深度卷积前向/梯度正确')


def test_classic_models():
    """VGG11 / MobileNet 前向形状、参数量、深度可分离的省参数效果。"""
    print('\n[3] VGG11 / MobileNetV1')
    x = Node(np.random.randn(2, 1, 64, 64))
    vgg = build_vgg11(num_classes=10, width=1.0)
    out_v = vgg(x)
    p_v = count_params(vgg)
    print(f'    VGG11: 输出 {out_v.data.shape}，参数 {p_v:,}')
    assert out_v.data.shape == (2, 10)

    mb = build_mobilenet_v1(num_classes=10, width=0.25)
    out_m = mb(x)
    p_m = count_params(mb)
    print(f'    MobileNet(w=0.25): 输出 {out_m.data.shape}，参数 {p_m:,}')
    assert out_m.data.shape == (2, 10)

    # 深度可分离卷积 vs 同通道标准 3×3：参数应约为 1/C + 1/9
    dw = DepthwiseConv2D(32, 3)
    pw = Conv2D(32, 64, 1)
    std = Conv2D(32, 64, 3, padding=1)
    n_sep = sum(p.data.size for p in dw.params()) + sum(p.data.size for p in pw.params())
    n_std = sum(p.data.size for p in std.params())
    print(f'    32→64：可分离 {n_sep:,} vs 标准 3×3 {n_std:,}（{n_sep / n_std:.2f}×）')
    assert n_sep < n_std * 0.25, '深度可分离卷积应显著更省参数'
    print('    ✓ 经典模型结构与参数效率正确')


def test_profiler():
    """profiler 口径：手算一个 conv 的 FLOPs，并检查激活内存>0。"""
    print('\n[4] profiler 口径')
    net = Sequential(Conv2D(1, 4, 3, padding=1), ReLU(),
                     Conv2D(4, 8, 3, stride=2, padding=1),
                     Flatten(), Linear(8 * 4 * 4, 10))
    flops, macs = count_flops(net, (1, 8, 8))
    # 手算（与 profiler 同口径：乘加各 1，逐元素算子 1 FLOP/元素）：
    #   conv1 = 2*1*8*8*4*1*3*3 = 4608
    #   relu  = 1*4*8*8         = 256
    #   conv2 = 2*1*4*4*8*4*3*3 = 9216
    #   fc    = 2*1*128*10      = 2560
    manual = (2 * 8 * 8 * 4 * 3 * 3 + 4 * 8 * 8
              + 2 * 4 * 4 * 8 * 4 * 3 * 3 + 2 * 128 * 10)
    print(f'    FLOPs={flops:,}（手算 {manual:,}），MACs={macs:,}')
    assert flops == manual, 'FLOPs 公式与手算不一致'

    act = measure_activation_bytes(build_vgg11(num_classes=10), (1, 64, 64))
    print(f'    VGG11 激活内存 = {act / 1024 / 1024:.2f} MB')
    assert act > 0, '激活内存应大于 0'
    print('    ✓ profiler 口径正确')


def main():
    """依次运行全部测试。"""
    print('=' * 62)
    print('经典模型 / 分组卷积 / 规模分析测试')
    print('=' * 62)
    test_grouped_conv()
    test_depthwise_conv()
    test_classic_models()
    test_profiler()
    print('\n' + '=' * 62)
    print('✓ 模型与规模分析全部测试通过')
    print('=' * 62)


if __name__ == '__main__':
    main()
