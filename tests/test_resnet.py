# -*- coding: utf-8 -*-
"""
test_resnet.py —— ResNet / GroupNorm 组件测试
================================================

覆盖：
    1. GroupNorm2D 的分组统计性质（均值≈0、方差≈1）；
    2. GroupNorm2D 的数值梯度；
    3. 极小 ResNet 的前向形状与全参数梯度；
    4. 默认 ResNet 的参数量（44,337）与保存/加载往返一致。

运行（项目根目录）：
    python tests/test_resnet.py
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

from miniflows import ops
from miniflows.graph import Node, param
from miniflows.losses import MSELoss
from miniflows.resnet import GroupNorm2D, build_resnet
from miniflows.autodiff import gradient_check, check_model_gradients
from miniflows.utils import save_model, load_model


def test_groupnorm_stats():
    """GroupNorm2D：每组内 (C/G,H,W) 归一化后均值≈0、方差≈1。"""
    print('\n[1] GroupNorm2D 分组统计')
    gn = GroupNorm2D(4, num_groups=2, eps=0.0)
    x = np.random.randn(2, 4, 8, 8) * 3 + 1
    y = gn(Node(x)).data.reshape(2, 2, 2, 8, 8)      # (N,G,C/G,H,W)
    mean_err = float(np.max(np.abs(y.mean(axis=(2, 3, 4)))))
    var_err = float(np.max(np.abs(y.var(axis=(2, 3, 4)) - 1.0)))
    print(f'    最大|组均值|={mean_err:.2e}，最大|组方差-1|={var_err:.2e}')
    assert mean_err < 1e-10 and var_err < 1e-10
    print('    ✓ GroupNorm 分组归一化正确')


def test_groupnorm_grad():
    """GroupNorm2D 的解析梯度 vs 中心差分。"""
    print('\n[2] GroupNorm2D 梯度')
    np.random.seed(0)
    gn = GroupNorm2D(4, num_groups=2)
    W = np.random.randn(2, 4, 8, 8)
    X = np.random.randn(2, 4, 8, 8)
    res = gradient_check(lambda ns: ops.sum(ops.mul(gn(ns[0]), W)), [X])
    print(f"    max_rel={res['max_rel_error']:.3e}  passed={res['passed']}")
    assert res['passed']
    print('    ✓ GroupNorm 梯度正确')


def test_tiny_resnet():
    """极小 ResNet：前向形状 + 全参数梯度。"""
    print('\n[3] 极小 ResNet 前向与全参数梯度')
    np.random.seed(0)
    m = build_resnet(c0=4, widths=(4, 8), blocks=(1, 1), num_groups=4)
    X = np.random.randn(2, 1, 16, 16)
    out = m(Node(X))
    print(f'    输入 {X.shape} -> 输出 {out.data.shape}，参数 {sum(p.data.size for p in m.params())}')
    assert out.data.shape == (2, 1)
    y = np.random.RandomState(1).randn(2, 1)
    res = check_model_gradients(m, X, MSELoss(), y=y)
    print(f"    梯度 passed={res['passed']} max_rel={res['max_rel_error']:.3e}")
    assert res['passed']
    print('    ✓ 极小 ResNet 前向/梯度正确')


def test_default_resnet():
    """默认 ResNet：参数量、64×64 前向、保存/加载往返。"""
    print('\n[4] 默认 ResNet 参数量与保存/加载')
    np.random.seed(1)
    m = build_resnet()
    n = sum(p.data.size for p in m.params())
    print(f'    参数量={n:,}（CNN 44,426 / ViT 44,353）')
    X = np.random.randn(3, 1, 64, 64)
    m.train_mode(False)
    p0 = m(Node(X)).data.copy()
    path = os.path.join(_ROOT, 'outputs', '_resnet_test.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_model(m, path)
    fresh = build_resnet()
    load_model(fresh, path)
    fresh.train_mode(False)
    diff = float(np.max(np.abs(p0 - fresh(Node(X)).data)))
    print(f'    保存/加载预测最大差={diff:.2e}')
    assert n == 44337, f'参数量应为 44337，实际 {n}'
    assert diff == 0.0
    os.remove(path)
    print('    ✓ 默认 ResNet 参数量/保存加载正确')


def main():
    """依次执行全部测试。"""
    print('=' * 62)
    print('ResNet / GroupNorm 组件测试')
    print('=' * 62)
    test_groupnorm_stats()
    test_groupnorm_grad()
    test_tiny_resnet()
    test_default_resnet()
    print('\n' + '=' * 62)
    print('✓ ResNet 全部测试通过')
    print('=' * 62)


if __name__ == '__main__':
    main()
