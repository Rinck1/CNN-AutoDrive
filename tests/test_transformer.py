# -*- coding: utf-8 -*-
"""
test_transformer.py —— ViT / Transformer 组件单元测试
=====================================================

覆盖：
    1. 新增算子 gelu / concat / broadcast_to / select 的数值梯度检验；
    2. 4D 批量 matmul（多头注意力用）的梯度检验；
    3. 极小 ViT 的前向形状、注意力权重形状、全参数梯度检验；
    4. LayerNorm 的归一化统计性质（均值≈0、方差≈1）；
    5. 模型保存/加载往返预测一致。

运行（项目根目录）：
    python tests/test_transformer.py
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
from miniflows.transformer import build_vit, LayerNorm
from miniflows.autodiff import gradient_check, check_model_gradients
from miniflows.utils import save_model, load_model


def test_new_ops_gradients():
    """对 ViT 新增的四个算子逐个做中心差分梯度检验。"""
    print('\n[1] 新增算子梯度检验')
    cases = []

    # GELU：随机输入，避开极小值区域（GELU 处处可导，但小梯度会放大相对误差）
    Xg = np.random.randn(3, 5) * 1.5
    cases.append(('gelu', lambda ns: ops.sum(ops.gelu(ns[0])), [Xg]))

    # concat：两个 (3,4) 沿 axis=1 拼接，加权求和让每个元素都有非平凡梯度
    A = np.random.randn(3, 4); B = np.random.randn(3, 4)
    Wc = np.random.randn(3, 8)
    cases.append(('concat', lambda ns: ops.sum(ops.mul(ops.concat([ns[0], ns[1]], axis=1),
                                                         Wc)), [A, B]))

    # broadcast_to：(2,1,3) -> (4,2,1,3)，反向必须沿扩展轴求和
    Xb = np.random.randn(2, 1, 3); Wb = np.random.randn(4, 2, 1, 3)
    cases.append(('broadcast_to',
                  lambda ns: ops.sum(ops.mul(ops.broadcast_to(ns[0], (4, 2, 1, 3)), Wb)),
                  [Xb]))

    # select：沿 axis=1 取第 2 个切片，加权求和，检验梯度是否只落在该切片
    Xs = np.random.randn(2, 5, 3); Ws = np.random.randn(2, 3)
    cases.append(('select', lambda ns: ops.sum(ops.mul(ops.select(ns[0], 2, axis=1), Ws)),
                  [Xs]))

    # 4D 批量 matmul：模拟多头注意力 (N,H,L,Dh) @ (N,H,Dh,L)
    Q = np.random.randn(2, 3, 4, 5); K = np.random.randn(2, 3, 5, 4)
    Wm = np.random.randn(2, 3, 4, 4)
    cases.append(('batched_matmul',
                  lambda ns: ops.sum(ops.mul(ops.matmul(ns[0], ns[1]), Wm)), [Q, K]))

    all_ok = True
    for name, fn, arrays in cases:
        res = gradient_check(fn, arrays, eps=1e-5, tol=1e-5)
        ok = res['passed']
        all_ok &= ok
        print(f"    {name:<18s} max_rel={res['max_rel_error']:.3e}  "
              f"{'✓' if ok else '✗'}")
    assert all_ok, '新增算子梯度检验失败'
    print('    ✓ 全部新增算子通过')


def test_layernorm_stats():
    """LayerNorm 前向：归一化后最后一维均值≈0、方差≈1（γ=1,β=0 初始）。"""
    print('\n[2] LayerNorm 统计性质')
    ln = LayerNorm(16, eps=0.0)   # eps=0：验证纯归一化（默认 eps 会带来 1e-6 级偏差）
    x = np.random.randn(4, 7, 16) * 3.0 + 2.0     # 故意用非零均值大方差
    y = ln(Node(x)).data
    mean_err = float(np.max(np.abs(y.mean(axis=-1))))
    var_err = float(np.max(np.abs(y.var(axis=-1) - 1.0)))
    print(f'    最大 |均值|={mean_err:.3e}，最大 |方差-1|={var_err:.3e}')
    assert mean_err < 1e-10 and var_err < 1e-10, 'LayerNorm 归一化统计不正确'
    print('    ✓ LayerNorm 归一化正确')


def test_tiny_vit():
    """极小 ViT：前向形状 + 注意力形状 + 全参数梯度检验。"""
    print('\n[3] 极小 ViT 前向与全参数梯度')
    np.random.seed(0)
    model = build_vit(img_size=16, patch_size=8, dim=16, depth=1, heads=2, mlp_ratio=2)
    X = np.random.randn(2, 1, 16, 16)
    out = model(Node(X))
    print(f'    输入 {X.shape} -> 输出 {out.data.shape}，参数 {sum(p.data.size for p in model.params())}')
    assert out.data.shape == (2, 1), 'ViT 输出形状不正确'
    attn = model.attention_maps(0)
    print(f'    注意力权重形状 {attn.shape}（应为 (N,H,L+1,L+1)）')
    assert attn.shape == (2, 2, 5, 5), '注意力权重形状不正确'

    y = np.random.RandomState(1).randn(2, 1)
    res = check_model_gradients(model, X, MSELoss(), y=y)
    print(f"    全参数梯度: passed={res['passed']}, max_rel={res['max_rel_error']:.3e}")
    assert res['passed'], 'ViT 全参数梯度检验失败'
    print('    ✓ 极小 ViT 前向/注意力/梯度全部正确')


def test_save_load_vit():
    """保存/加载 ViT 参数，预测应逐位一致。"""
    print('\n[4] ViT 保存/加载往返')
    np.random.seed(1)
    model = build_vit(img_size=16, patch_size=8, dim=16, depth=1, heads=2)
    X = np.random.randn(4, 1, 16, 16)
    model.train_mode(False)
    p0 = model(Node(X)).data.copy()
    path = os.path.join(_ROOT, 'outputs', '_vit_test_model.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_model(model, path)
    fresh = build_vit(img_size=16, patch_size=8, dim=16, depth=1, heads=2)
    load_model(fresh, path)
    fresh.train_mode(False)
    p1 = fresh(Node(X)).data
    diff = float(np.max(np.abs(p0 - p1)))
    print(f'    保存/加载预测最大差 = {diff:.2e}')
    assert diff == 0.0, 'ViT 保存/加载不一致'
    os.remove(path)
    print('    ✓ ViT 保存/加载一致')


def main():
    """依次执行全部测试，任何断言失败都会抛出异常。"""
    print('=' * 62)
    print('ViT / Transformer 组件单元测试')
    print('=' * 62)
    test_new_ops_gradients()
    test_layernorm_stats()
    test_tiny_vit()
    test_save_load_vit()
    print('\n' + '=' * 62)
    print('✓ Transformer 全部测试通过')
    print('=' * 62)


if __name__ == '__main__':
    main()
