# -*- coding: utf-8 -*-
"""
tests/test_core.py —— MiniFlows 核心框架自测（算子形状 / 数值梯度检验 / 训练 / CNN / 其他）
==========================================================================================

【测试内容】
    1. Node 基础：运算符混算（标量/ndarray）、梯度累加（节点复用）、
       detach、build_topo / collect_edges、常量不积累梯度；
    2. 全算子前向形状 + 正确性（conv2d/pool 与朴素循环参考实现比对，
       含 stride/padding 各种组合）；im2col/col2im 往返一致性；
    3. gradient_check 对全部算子做数值梯度检验（相对误差 < 1e-5，float64）；
    4. MLP(2-8-1) 合成回归：MBGD 与 Adam 分别训练，loss 明显下降；
    5. 小 CNN：前向形状 + check_model_gradients 全参数检验；
    6. Sequential.summary() 中文表格 + Dropout 训练/评估行为 + named_params 命名。

【运行方式】（项目根目录）
    python tests/test_core.py
    全部通过打印：✓ 核心框架全部测试通过
"""

import os
import sys
import time
import traceback

# ---------- 标准初始化头：Windows 控制台 UTF-8 + 项目根目录入 sys.path ----------
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from miniflows.graph import Node, constant, param, build_topo, collect_edges
from miniflows import ops
from miniflows.layers import (Layer, Linear, Conv2D, MaxPool2D, AvgPool2D,
                              Flatten, ReLU, Sigmoid, Tanh, Softmax,
                              Dropout, Sequential)
from miniflows.losses import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss, l2_penalty
from miniflows.optim import Optimizer, MBGD, Adam
from miniflows.autodiff import numerical_gradient, gradient_check, check_model_gradients


# ============================================================================
#                        测试 1：Node 基础行为
# ============================================================================
def test_node_basic():
    """测试 Node：运算符混算、梯度累加、detach、图工具、常量行为。"""
    np.random.seed(0)
    x = param(np.array([[1.0, 2.0], [3.0, 4.0]]), name='x')

    # ---- 1) 运算符重载与标量/ndarray 混算（自动包装常量）----
    assert np.allclose((x + 1.5).data, x.data + 1.5), 'Node + 标量'
    assert np.allclose((2.5 + x).data, 2.5 + x.data), '标量 + Node (__radd__)'
    assert np.allclose((x - np.ones((2, 2))).data, x.data - 1.0), 'Node - ndarray'
    assert np.allclose((np.ones((2, 2)) - x).data, 1.0 - x.data), 'ndarray - Node (__rsub__)'
    assert np.allclose((x * 3).data, x.data * 3), 'Node * 标量'
    assert np.allclose((0.5 * x).data, 0.5 * x.data), '标量 * Node (__rmul__)'
    assert np.allclose((x / 2).data, x.data / 2), 'Node / 标量'
    assert np.allclose((12.0 / x).data, 12.0 / x.data), '标量 / Node (__rtruediv__)'
    assert np.allclose((-x).data, -x.data), '取负 (__neg__)'
    assert np.allclose((x ** 2).data, x.data ** 2), '幂 (__pow__)'
    assert np.allclose((x @ np.ones((2, 1))).data, x.data @ np.ones((2, 1))), '矩阵乘 (__matmul__)'

    # ---- 2) 梯度累加：x 在一个式子里被使用两次 ----
    # y = x*x + x => dy/dx = 2x + 1（若 += 累加实现有误，这里只会得到其中一份）
    x.zero_grad()
    y = x * x + x
    y.backward()
    assert np.allclose(x.grad, 2 * x.data + 1), \
        f'梯度累加失败: {x.grad} != {2 * x.data + 1}'

    # ---- 3) detach：脱离计算图（无父节点、requires_grad=False）----
    d = x.detach()
    assert d.requires_grad is False and d.parents == []
    assert np.allclose(d.data, x.data)

    # ---- 4) 常量节点不积累梯度，但参与前向 ----
    c = constant(2.0)
    s = (c * x).sum()
    s.backward()
    assert c.grad is None, '常量节点不应有梯度'
    assert np.allclose(x.grad, 2.0 * np.ones((2, 2))), 'c*x 对 x 的梯度应为 2'

    # ---- 5) build_topo / collect_edges ----
    a = param(np.ones(2), name='a')
    b = param(np.ones(2), name='b')
    z = a * b + a          # 节点: a, b, mul, z(add) —— a 被使用两次
    topo = build_topo(z)
    assert topo[-1] is z and len(topo) == 4, 'build_topo 应含 4 个节点且 root 在最后'
    edges = collect_edges(z)
    assert len(edges) == 4, f'collect_edges 应有 4 条边，得到 {len(edges)}'
    # 拓扑序合法性：每个节点都在其父节点之后出现
    pos = {id(n): i for i, n in enumerate(topo)}
    for n in topo:
        for p in n.parents:
            assert pos[id(p)] < pos[id(n)], '拓扑序违反：父节点必须在前'
    z.backward()
    assert np.allclose(b.grad, np.ones(2)), 'z=a*b+a 对 b 的梯度应为 1'

    # ---- 6) float64 统一 + shape 属性 + zero_grad ----
    n_mixed = Node(np.arange(6).reshape(2, 3).astype(np.float32))  # 故意给 float32
    assert n_mixed.data.dtype == np.float64, 'Node 内部必须统一 float64'
    assert n_mixed.shape == (2, 3)
    x.zero_grad()
    assert x.grad is None
    print('  [1] Node 基础（运算符混算/梯度累加/detach/图工具） —— 通过')


# ============================================================================
#                测试 2：全算子前向形状与正确性（含参考实现比对）
# ============================================================================
def naive_conv2d(x, W, b, stride, padding):
    """朴素六重循环卷积参考实现（用于与 im2col 版比对正确性）。"""
    N, C, H, Wd = x.shape
    F, _, KH, KW = W.shape
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (Wd + 2 * padding - KW) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.zeros((N, F, OH, OW))
    for n in range(N):
        for f in range(F):
            for i in range(OH):
                for j in range(OW):
                    patch = xp[n, :, i * stride:i * stride + KH,
                               j * stride:j * stride + KW]
                    out[n, f, i, j] = np.sum(patch * W[f]) + (b[f] if b is not None else 0.0)
    return out


def naive_maxpool(x, k, s):
    """朴素最大池化参考实现。"""
    N, C, H, W = x.shape
    OH, OW = (H - k) // s + 1, (W - k) // s + 1
    out = np.zeros((N, C, OH, OW))
    for i in range(OH):
        for j in range(OW):
            out[:, :, i, j] = x[:, :, i * s:i * s + k, j * s:j * s + k].max(axis=(2, 3))
    return out


def test_op_shapes():
    """测试各算子前向形状（含 conv/pool 的 stride/padding 组合）与数值正确性。"""
    np.random.seed(1)
    x4 = np.random.randn(2, 3, 8, 8)              # 通用 4 维输入

    # ---- 1) conv2d 各种 stride/padding/核大小组合 vs 朴素参考实现 ----
    combos = [(1, 0, 3), (2, 1, 3), (1, 2, 5), (3, 1, 3), (2, 0, 2)]
    for stride, padding, k in combos:
        Wt = np.random.randn(4, 3, k, k)
        bt = np.random.randn(4)
        got = ops.conv2d(Node(x4), Node(Wt), Node(bt),
                         stride=stride, padding=padding).data
        ref = naive_conv2d(x4, Wt, bt, stride, padding)
        assert got.shape == ref.shape, f'conv2d(s={stride},p={padding},k={k}) 形状不符'
        assert np.allclose(got, ref, atol=1e-10), \
            f'conv2d(s={stride},p={padding},k={k}) 与朴素实现数值不符'
        assert got.shape == (2, 4, (8 + 2 * padding - k) // stride + 1,
                             (8 + 2 * padding - k) // stride + 1)
    # 无偏置版本
    got = ops.conv2d(Node(x4), Node(np.random.randn(2, 3, 3, 3)),
                     stride=1, padding=1).data
    assert got.shape == (2, 2, 8, 8), 'conv2d 无偏置形状不符'

    # ---- 2) maxpool/avgpool 各种窗口/步幅（含奇数尺寸向下取整）----
    for k, s in [(2, 2), (3, 1), (3, 2), (2, 1)]:
        xin = np.random.randn(2, 3, 7, 7)
        got = ops.maxpool2d(Node(xin), kernel_size=k, stride=s).data
        ref = naive_maxpool(xin, k, s)
        assert got.shape == ref.shape == (2, 3, (7 - k) // s + 1, (7 - k) // s + 1)
        assert np.allclose(got, ref), f'maxpool(k={k},s={s}) 与朴素实现不符'
        got_avg = ops.avgpool2d(Node(xin), kernel_size=k, stride=s).data
        assert got_avg.shape == got.shape, f'avgpool(k={k},s={s}) 形状不符'

    # ---- 3) 基础逐元素运算与广播 ----
    assert ops.add(Node(x4), Node(np.ones(8))).shape == (2, 3, 8, 8), 'add 广播'
    assert ops.sub(Node(x4), 1.0).shape == x4.shape, 'sub'
    assert ops.mul(Node(x4), Node(np.ones((3, 1)))).shape == x4.shape, 'mul 广播'
    assert ops.div(Node(x4), Node(2.0)).shape == x4.shape, 'div'
    assert ops.matmul(Node(np.random.randn(2, 3)), Node(np.random.randn(3, 4))).shape == (2, 4), 'matmul'
    assert ops.broadcast_add(Node(x4), Node(np.random.randn(3, 1))).shape == x4.shape, 'broadcast_add'
    assert ops.neg(Node(x4)).shape == x4.shape, 'neg'

    # ---- 4) 激活函数 ----
    z = np.random.randn(2, 3, 4, 5)
    assert ops.relu(Node(z)).shape == z.shape
    assert ops.sigmoid(Node(z)).shape == z.shape
    assert np.allclose(ops.sigmoid(Node(z)).data, 1.0 / (1.0 + np.exp(-z))), 'sigmoid 数值'
    assert ops.tanh(Node(z)).shape == z.shape
    sm = ops.softmax(Node(z), axis=-1).data
    assert sm.shape == z.shape and np.allclose(sm.sum(axis=-1), 1.0), 'softmax 归一性'

    # ---- 5) 形状/归约 ----
    assert ops.reshape(Node(x4), (2, -1)).shape == (2, 192), 'reshape'
    assert ops.flatten(Node(x4), start_axis=1).shape == (2, 192), 'flatten'
    assert ops.transpose(Node(x4), (1, 0, 3, 2)).shape == (3, 2, 8, 8), 'transpose'
    assert ops.sum(Node(x4), axis=1).shape == (2, 8, 8), 'sum axis'
    assert ops.sum(Node(x4)).shape == (), 'sum 全量 -> 标量'
    assert ops.mean(Node(x4), axis=-1, keepdims=True).shape == (2, 3, 8, 1), 'mean keepdims'
    assert float(ops.mean(Node(x4)).data) == x4.mean(), 'mean 全量数值'

    # ---- 6) im2col/col2im 往返一致（stride==kernel 的非重叠窗口可精确还原）----
    ximg = np.random.randn(2, 3, 6, 6)
    col = ops.im2col(ximg, 2, 2, 2, 0)
    assert col.shape == (2 * 3 * 3, 3 * 2 * 2), 'im2col 形状'
    back = ops.col2im(col, ximg.shape, 2, 2, 2, 0)
    assert np.allclose(back, ximg, atol=1e-12), 'im2col->col2im 非重叠窗口应精确还原'
    print('  [2] 全算子前向形状 + 朴素实现比对（conv/pool stride/padding 组合） —— 通过')


# ============================================================================
#                测试 3：gradient_check 全算子数值梯度检验
# ============================================================================
GC_RESULTS = {}    # 记录每个算子的检验结果，最后统一打印表格


def do_gc(name, f, arrays, eps=1e-5, tol=1e-5):
    """运行一次 gradient_check 并记录结果；失败立即断言报错。"""
    res = gradient_check(f, arrays, eps=eps, tol=tol)
    GC_RESULTS[name] = res
    assert res['passed'], \
        f'{name} 梯度检验失败: max_rel_error={res["max_rel_error"]:.3e} (tol={tol})'
    return res


def test_gradient_checks():
    """对契约要求的全部算子做数值梯度检验（相对误差 < 1e-5）。

    注意事项（都在具体用例中体现）：
        · relu/maxpool 等分段线性算子要避开"折点"（输入远离 0）；
        · maxpool 用叠加递增斜坡的随机输入保证窗口内最大值无平局；
        · div/log/pow 等要保证定义域合法（分母/真数远离 0）；
        · exp 控制输入幅度防溢出。
    """
    np.random.seed(42)

    # ---- add（含广播情形，同时检验 unbroadcast）----
    do_gc('add', lambda ns: ops.add(ns[0], ns[1]).sum(),
          [np.random.randn(3, 4), np.random.randn(3, 4)])
    do_gc('add广播', lambda ns: ops.add(ns[0], ns[1]).sum(),
          [np.random.randn(3, 4), np.random.randn(4)])

    # ---- sub / mul / div ----
    do_gc('sub', lambda ns: ops.sub(ns[0], ns[1]).sum(),
          [np.random.randn(3, 4), np.random.randn(3, 4)])
    do_gc('mul', lambda ns: ops.mul(ns[0], ns[1]).sum(),
          [np.random.randn(3, 4), np.random.randn(3, 4)])
    do_gc('div', lambda ns: ops.div(ns[0], ns[1]).sum(),
          [np.random.randn(3, 4), np.random.randn(3, 4) + 3.0])   # 分母远离 0

    # ---- matmul / broadcast_add ----
    do_gc('matmul', lambda ns: ops.matmul(ns[0], ns[1]).sum(),
          [np.random.randn(3, 4), np.random.randn(4, 2)])
    do_gc('broadcast_add', lambda ns: ops.broadcast_add(ns[0], ns[1]).sum(),
          [np.random.randn(2, 3, 4, 4), np.random.randn(1, 4, 1, 1)])  # (1,F,1,1) 偏置
    do_gc('broadcast_add向量', lambda ns: ops.broadcast_add(ns[0], ns[1]).sum(),
          [np.random.randn(2, 3, 4, 4), np.random.randn(4)])

    # ---- exp / log / pow / sqrt ----
    do_gc('exp', lambda ns: ops.exp(ns[0]).sum(), [np.random.randn(3, 4) * 0.5])  # 幅度小防溢出
    do_gc('log', lambda ns: ops.log(ns[0]).sum(),
          [np.abs(np.random.randn(3, 4)) + 0.5])                   # 真数 > 0
    do_gc('pow', lambda ns: ops.pow(ns[0], 3).sum(),
          [np.random.randn(3, 4) * 0.5 + 1.5])                     # 底数 > 0
    do_gc('sqrt', lambda ns: ops.sqrt(ns[0]).sum(),
          [np.abs(np.random.randn(3, 4)) + 0.5])

    # ---- 激活函数（relu 输入推离折点 |x|>=1e-3）----
    x_relu = np.random.randn(3, 4)
    x_relu[np.abs(x_relu) < 1e-3] = 0.5       # 把贴着 0 的元素推开，避免差分跨越折点
    do_gc('relu', lambda ns: ops.relu(ns[0]).sum(), [x_relu])
    do_gc('sigmoid', lambda ns: ops.sigmoid(ns[0]).sum(), [np.random.randn(3, 4)])
    do_gc('tanh', lambda ns: ops.tanh(ns[0]).sum(), [np.random.randn(3, 4)])
    do_gc('softmax', lambda ns: ops.softmax(ns[0], axis=-1).sum(), [np.random.randn(4, 5)])
    do_gc('softmax轴0', lambda ns: ops.softmax(ns[0], axis=0).sum(), [np.random.randn(4, 5)])

    # ---- conv2d（两组 stride/padding 配置，同时检验 x/W/b 三者梯度）----
    do_gc('conv2d(s1,p0)', lambda ns: ops.conv2d(ns[0], ns[1], ns[2],
                                                 stride=1, padding=0).sum(),
          [np.random.randn(2, 2, 5, 5), np.random.randn(3, 2, 3, 3), np.random.randn(3)])
    do_gc('conv2d(s2,p1)', lambda ns: ops.conv2d(ns[0], ns[1], ns[2],
                                                 stride=2, padding=1).sum(),
          [np.random.randn(1, 2, 6, 6), np.random.randn(2, 2, 3, 3), np.random.randn(2)])

    # ---- maxpool2d（随机 + 斜坡保证无重复值，避免窗口内最大值平局歧义）----
    def unique_rand(shape):
        """生成互不相同、且远离 0 的随机数组（斜坡打散重复 + 推离折点）。"""
        x = np.random.randn(*shape).ravel()
        x = x + np.arange(x.size) * 1e-3            # 递增斜坡：保证严格无重复
        x[np.abs(x) < 1e-3] += 5e-3                 # 推离 0，避免池化边界歧义
        return x.reshape(shape)
    do_gc('maxpool2d(k2,s2)', lambda ns: ops.maxpool2d(ns[0], 2, 2).sum(),
          [unique_rand((2, 3, 6, 6))])
    do_gc('maxpool2d(k3,s1)', lambda ns: ops.maxpool2d(ns[0], 3, 1).sum(),
          [unique_rand((2, 2, 5, 5))])              # 重叠窗口（stride<kernel）

    # ---- avgpool2d ----
    do_gc('avgpool2d(k2,s2)', lambda ns: ops.avgpool2d(ns[0], 2, 2).sum(),
          [np.random.randn(2, 2, 6, 6)])
    do_gc('avgpool2d(k3,s1)', lambda ns: ops.avgpool2d(ns[0], 3, 1).sum(),
          [np.random.randn(2, 2, 5, 5)])

    # ---- 形状/归约 ----
    do_gc('reshape', lambda ns: ops.reshape(ns[0], (3, 8)).sum(), [np.random.randn(2, 3, 4)])
    do_gc('transpose', lambda ns: ops.transpose(ns[0], (1, 0, 2)).sum(), [np.random.randn(2, 3, 4)])
    do_gc('flatten', lambda ns: ops.flatten(ns[0], 1).sum(), [np.random.randn(2, 3, 4)])
    do_gc('sum(axis=0)', lambda ns: ops.sum(ns[0], axis=0).sum(), [np.random.randn(3, 4)])
    do_gc('sum(全量)', lambda ns: ops.sum(ns[0]), [np.random.randn(3, 4)])
    do_gc('mean(axis=-1,keepdims)', lambda ns: ops.mean(ns[0], axis=-1, keepdims=True).sum(),
          [np.random.randn(3, 4)])
    do_gc('mean(全量)', lambda ns: ops.mean(ns[0]), [np.random.randn(3, 4)])

    # ---- 三个损失函数（目标/标签固定为常量，只检验对输入的梯度）----
    do_gc('MSELoss', lambda ns: MSELoss()(ns[0], np.random.RandomState(3).randn(4, 3)),
          [np.random.randn(4, 3)])
    do_gc('BCEWithLogitsLoss', lambda ns: BCEWithLogitsLoss()(ns[0], np.array([0., 1., 1., 0., 1., 0.])),
          [np.random.randn(6)])
    do_gc('BCEWithLogitsLoss(N,1)', lambda ns: BCEWithLogitsLoss()(ns[0], np.array([[0.], [1.], [1.]])),
          [np.random.randn(3, 1)])
    do_gc('CrossEntropyLoss', lambda ns: CrossEntropyLoss()(ns[0], np.array([1, 3, 0, 2, 2])),
          [np.random.randn(5, 4)])

    # ---- 汇总打印检验结果表 ----
    print('  [3] 全算子数值梯度检验（梯度检验报告）：')
    worst = 0.0
    for name, res in GC_RESULTS.items():
        worst = max(worst, res['max_rel_error'])
        print(f'      {name:<26} max_rel_error = {res["max_rel_error"]:.3e}   ✓')
    print(f'      >>> 全部 {len(GC_RESULTS)} 项通过，全局最大相对误差 = {worst:.3e} (< 1e-5)')
    assert worst < 1e-5


# ============================================================================
#                测试 4：MLP(2-8-1) 合成回归训练（MBGD 与 Adam）
# ============================================================================
def _make_regression_data(n=256, seed=0):
    """构造合成回归数据：y = sin(1.5·x1) + 0.5·cos(2·x2)（非线性可分）。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = np.sin(1.5 * X[:, :1]) + 0.5 * np.cos(2.0 * X[:, 1:2])
    return X, y


def _train_mlp(optimizer_kind):
    """用指定优化器训练 MLP(2-8-1)，返回 (初始loss, 最终loss, 模型)。"""
    X, y = _make_regression_data(n=256, seed=7)
    # 2 输入 -> 8 隐藏（Tanh）-> 1 输出；隐藏层用 xavier 初始化配合 tanh
    model = Sequential(Linear(2, 8, init='xavier'), Tanh(),
                       Linear(8, 1, init='xavier'))
    loss_fn = MSELoss()
    if optimizer_kind == 'MBGD':
        opt = MBGD(model.params(), lr=0.1, momentum=0.9)   # 动量版小批量梯度下降
    else:
        opt = Adam(model.params(), lr=0.01)                # Adam 自适应学习率
    epochs, batch = 300, 32
    n = X.shape[0]
    first_loss = None
    for epoch in range(epochs):
        # 每个 epoch 打乱数据（固定种子的置换流，保证可复现）
        perm = np.random.permutation(n)
        model.train_mode(True)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            xb, yb = X[idx], y[idx]
            pred = model(Node(xb))                # 前向
            loss = loss_fn(pred, yb)              # MSE 损失（标量 Node）
            opt.zero_grad()                       # 清梯度
            loss.backward()                       # 反向传播
            opt.step()                            # 更新参数（直接改 p.data）
            if first_loss is None:
                first_loss = float(loss.data)     # 记录首个 batch 的损失
    # 训练结束全量损失
    model.train_mode(False)
    final_loss = float(loss_fn(model(Node(X)), y).data)
    return first_loss, final_loss, model


def test_mlp_training():
    """MLP(2-8-1) 在合成回归数据上分别用 MBGD 和 Adam 训练，loss 必须明显下降。"""
    print('  [4] MLP(2-8-1) 合成回归训练：')
    for kind in ['MBGD', 'Adam']:
        np.random.seed(42)
        l0, l1, _ = _train_mlp(kind)
        print(f'      {kind:<5} 初始 loss = {l0:.5f}  ->  训练后 loss = {l1:.5f}'
              f'  (下降到 {(l1 / l0 * 100):.1f}%)')
        # 判据：最终损失至少降到初始的 20% 以下（"明显下降"），且绝对值足够小
        assert l1 < 0.2 * l0, f'{kind} 训练后 loss 未明显下降: {l0:.5f} -> {l1:.5f}'
        assert l1 < 0.05, f'{kind} 训练后 loss 偏高: {l1:.5f}'


# ============================================================================
#                测试 5：小 CNN（Conv+ReLU+MaxPool+Flatten+Linear）
# ============================================================================
def test_cnn():
    """小 CNN 前向形状正确 + check_model_gradients 全参数数值检验通过。"""
    np.random.seed(7)
    # Conv(1->2,3,pad1) -> ReLU -> MaxPool2 -> Flatten -> Linear(32->3)
    model = Sequential(
        Conv2D(1, 2, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Flatten(),
        Linear(2 * 4 * 4, 3),
    )
    x = np.random.randn(2, 1, 8, 8)
    pred = model(Node(x))
    # (2,1,8,8) --conv(pad1)--> (2,2,8,8) --pool2--> (2,2,4,4) --flatten--> (2,32) --linear--> (2,3)
    assert pred.data.shape == (2, 3), f'CNN 前向形状错误: {pred.data.shape} != (2, 3)'

    # ---- 全参数数值梯度检验（MSE 损失，固定随机目标）----
    y = np.random.RandomState(1).randn(2, 3)
    res = check_model_gradients(model, x, MSELoss(), y=y)
    n_params = len(res['per_param'])
    print(f'  [5] 小 CNN 全参数梯度检验：{n_params} 个参数（W/b×4 层），'
          f'passed={res["passed"]}, max_rel_error={res["max_rel_error"]:.3e}')
    for pname, info in res['per_param'].items():
        print(f'      {pname:<14} shape={info["shape"]}  '
              f'max_rel_error={info["max_rel_error"]:.3e}')
    assert res['passed'], f'CNN 全参数梯度检验失败: {res["max_rel_error"]:.3e}'

    # ---- 附带检验：l2_penalty 与损失相加后梯度依然正确 ----
    np.random.seed(8)
    model2 = Sequential(Conv2D(1, 2, 3, padding=1), ReLU(), Flatten(), Linear(128, 2))
    x2 = np.random.randn(2, 1, 6, 6)
    y2 = np.random.RandomState(2).randn(2, 2)
    loss_fn = MSELoss()
    res2 = check_model_gradients(
        model2, x2,
        lambda pred, tgt: loss_fn(pred, tgt) + l2_penalty(model2, 0.01),  # loss + L2 组合
        y=y2)
    print(f'      [附带] loss + l2_penalty(λ=0.01) 组合检验：passed={res2["passed"]}, '
          f'max_rel_error={res2["max_rel_error"]:.3e}')
    assert res2['passed'], 'loss + l2_penalty 组合梯度检验失败'


# ============================================================================
#          测试 6：Sequential.summary() / Dropout 行为 / named_params
# ============================================================================
def test_summary_and_dropout():
    """测试 summary 中文表格、Dropout 训练/评估行为、named_params 命名与 zero_grad。"""
    np.random.seed(3)

    # ---- 1) summary：打印中文表格并校验总参数量 ----
    model = Sequential(
        Conv2D(1, 4, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Flatten(),
        Linear(4 * 4 * 4, 10),
        ReLU(),
        Linear(10, 2),
    )
    print('  [6] Sequential.summary() 输出：')
    total = model.summary((1, 8, 8))
    manual = sum(p.data.size for p in model.params())
    assert total == manual, f'summary 总参数量 {total} != 手工统计 {manual}'
    print(f'      summary 返回总参数量 = {total}，与手工统计一致 ✓')

    # ---- 2) Dropout 训练/评估行为 ----
    drop = Dropout(p=0.5)
    xin = np.random.randn(400, 50) + 10.0      # 全体为正，便于按值识别保留位
    # 训练模式：约一半置零，保留位 = 原值/(1-p) = 2 倍（inverted dropout）
    drop.train_mode(True)
    out_train = drop(Node(xin)).data
    zero_ratio = float(np.mean(out_train == 0.0))
    kept = out_train != 0.0
    assert 0.4 < zero_ratio < 0.6, f'Dropout 置零比例异常: {zero_ratio:.3f}'
    assert np.allclose(out_train[kept], 2.0 * xin[kept]), 'Dropout 保留位未按 1/(1-p) 缩放'
    mean_ratio = float(out_train.mean() / xin.mean())     # 期望应与输入一致
    assert 0.8 < mean_ratio < 1.2, f'Dropout 期望补偿失败: 均值比 {mean_ratio:.3f}'
    # 评估模式：恒等映射
    drop.train_mode(False)
    out_eval = drop(Node(xin)).data
    assert np.allclose(out_eval, xin), 'Dropout 评估模式必须是恒等映射'
    print(f'      Dropout(p=0.5): 训练态置零比例={zero_ratio:.3f}, '
          f'保留位缩放=2.0, 均值比={mean_ratio:.3f}；评估态恒等 ✓')

    # ---- 3) Sequential 内 Dropout 的 train_mode 递归生效 ----
    seq_d = Sequential(Linear(4, 4), Dropout(0.5))
    seq_d.train_mode(False)
    xd = np.random.randn(3, 4)
    assert np.allclose(seq_d(Node(xd)).data, seq_d.layers[0](Node(xd)).data), \
        'Sequential.train_mode(False) 未递归关掉 Dropout'
    seq_d.train_mode(True)
    assert not np.allclose(seq_d(Node(xd)).data, seq_d.layers[0](Node(xd)).data), \
        'Sequential.train_mode(True) 后 Dropout 未生效'
    print('      Sequential.train_mode 递归生效 ✓')

    # ---- 4) named_params 命名规则 + zero_grad + 参数保存名唯一性 ----
    named_conv = Sequential(Conv2D(2, 2, 3, name='conv1'), Flatten(), Linear(18, 2))
    names = [n for n, _ in named_conv.named_params()]
    assert names == ['conv1.W', 'conv1.b', 'layer1.W', 'layer1.b'], \
        f'named_params 命名错误: {names}'
    named_conv.zero_grad()
    assert all(p.grad is None for _, p in named_conv.named_params()), 'zero_grad 未清空'
    assert len(named_conv.params()) == 4
    # Layer.params/named_params 默认空实现（无参数层）
    assert ReLU().params() == [] and Flatten().named_params() == []
    print(f'      named_params 命名 {names} ✓；zero_grad / 无参数层默认接口 ✓')

    # ---- 5) 优化器基本行为（step 后参数确实变化；lr 可被调度器修改）----
    np.random.seed(4)
    pnode = param(np.ones(3))
    opt = MBGD([pnode], lr=0.1)
    y = (pnode * pnode).sum()      # L = Σp²，梯度 2p
    y.backward()
    old = pnode.data.copy()
    opt.step()
    assert np.allclose(pnode.data, old - 0.1 * 2 * old), 'MBGD.step 更新错误'
    opt.lr = 0.05                  # 模拟学习率调度器直接修改 lr
    assert opt.lr == 0.05
    print('      MBGD.step 更新与 lr 可调 ✓')


# ============================================================================
#                                  主入口
# ============================================================================
def main():
    """依次运行全部测试；任何一项失败即打印堆栈并以非零码退出。"""
    np.random.seed(42)     # 全局种子兜底（各测试内部还会重设，保证可复现）
    tests = [
        ('测试1 Node 基础行为', test_node_basic),
        ('测试2 全算子前向形状与正确性', test_op_shapes),
        ('测试3 全算子数值梯度检验', test_gradient_checks),
        ('测试4 MLP 训练（MBGD/Adam）', test_mlp_training),
        ('测试5 小 CNN 梯度检验', test_cnn),
        ('测试6 summary/Dropout/其他', test_summary_and_dropout),
    ]
    t0 = time.time()
    for title, fn in tests:
        print(f'\n===== {title} =====')
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f'✗ {title} 失败')
            sys.exit(1)
    print(f'\n总耗时 {time.time() - t0:.1f} s')
    print('✓ 核心框架全部测试通过')


if __name__ == '__main__':
    main()
