# -*- coding: utf-8 -*-
"""
miniflows/autodiff.py —— 数值梯度与梯度检验工具（扩展②）
=========================================================

【模块定位】
    自动微分写完后必须"验算"。本模块提供两类工具：
        1) numerical_gradient: 对任意"数组列表 -> 标量"的函数做中心差分；
        2) gradient_check:     同一函数分别用【解析梯度(backward)】与
           【数值梯度(中心差分)】计算，逐元素比较相对误差；
        3) check_model_gradients: 对完整模型的全部参数做上述检验。

【数学原理：中心差分数值梯度】
    对标量函数 f 在 x_i 处的偏导，用中心差分近似：
        df/dx_i ≈ [f(x_i + ε) - f(x_i - ε)] / (2ε)
    误差阶为 O(ε²)（前向差分是 O(ε)），float64 + ε=1e-5 时可达 ~1e-10 精度，
    足以暴露反向传播实现中绝大多数推导错误。

【相对误差判据（契约公式）】
        rel = |a - n| / max(1e-8, |a| + |n|)
    其中 a 为解析梯度、n 为数值梯度。补充设计：当 |a| 与 |n| 同时 < 1e-8
    （真值就是 0、数值结果只是 ~1e-11 的差分噪声，如 ReLU 负区/最大池化
    非最大位置）时，该元素误差记 0 —— 否则 1e-11/1e-8=1e-3 的伪误差
    会让所有"梯度天然为零"的算子永远无法通过 tol=1e-5 的检验。

【依赖】numpy + miniflows.graph。
"""

import numpy as np

from miniflows.graph import Node


# ============================================================================
#                          数值梯度（中心差分）
# ============================================================================
def numerical_gradient(f, arrays, eps=1e-5):
    """对函数 f 做"逐元素中心差分"数值梯度。

    参数：
        f: 接收 list[np.ndarray]、返回标量 float 的函数；
        arrays (list[np.ndarray]): 待求梯度的数组列表（会被复制，不污染原数据）；
        eps (float): 中心差分步长。

    返回：
        list[np.ndarray]: 与 arrays 一一对应的数值梯度列表。

    实现思路：
        对每个数组的每个元素：+ε 算一次 f、-ε 算一次 f，二者差除以 2ε；
        复杂度 O(总元素数 × 2 次前向)，只用于离线检验、不用于训练。
    """
    # 复制一份，避免扰动调用方的原始数据（也保证元素可写）
    arrays = [np.asarray(a, dtype=np.float64).copy() for a in arrays]
    grads = [np.zeros_like(a) for a in arrays]
    for k, arr in enumerate(arrays):
        it = np.nditer(arr, flags=['multi_index'])   # 通用 N 维迭代器
        while not it.finished:
            idx = it.multi_index                     # 当前元素的多维索引
            orig = arr[idx]
            arr[idx] = orig + eps                    # x + ε
            f_plus = float(f(arrays))
            arr[idx] = orig - eps                    # x - ε
            f_minus = float(f(arrays))
            arr[idx] = orig                          # 恢复原值
            # 中心差分：误差 O(ε²)，比前向差分精确得多
            grads[k][idx] = (f_plus - f_minus) / (2.0 * eps)
            it.iternext()
    return grads


# ============================================================================
#                          相对误差小工具（私有）
# ============================================================================
def _rel_error(analytic, numeric):
    """逐元素相对误差 rel = |a-n| / max(1e-8, |a|+|n|)（契约公式）。

    补充设计（见模块头说明）：|a|、|n| 同时 < 1e-8 的元素（梯度真值为 0、
    数值结果只是差分噪声）误差记 0，避免伪误差。
    """
    a = np.asarray(analytic, dtype=np.float64)
    n = np.asarray(numeric, dtype=np.float64)
    denom = np.maximum(1e-8, np.abs(a) + np.abs(n))   # 契约规定的分母下限
    rel = np.abs(a - n) / denom
    near_zero = (np.abs(a) < 1e-8) & (np.abs(n) < 1e-8)   # 双零（噪声主导）区
    return np.where(near_zero, 0.0, rel)


def _max_rel(analytic, numeric):
    """返回一对梯度的最大相对误差（float）。"""
    if analytic is None:      # 参数没拿到梯度（未参与前向），视为完美通过
        return 0.0
    return float(_rel_error(analytic, numeric).max())


# ============================================================================
#                     算子级梯度检验（封装成三步流程）
# ============================================================================
def gradient_check(f, arrays, eps=1e-5, tol=1e-5):
    """对一个"输入 Node 列表 -> 标量 Node"的函数做完整梯度检验。

    参数：
        f: 接收 list[Node]、返回标量 Node 的函数（通常是 某算子+sum 归约
           或 某损失函数）；
        arrays (list[np.ndarray]): 与 f 输入对应的数组；
        eps (float): 数值差分步长；
        tol (float): 通过阈值（所有输入的最大相对误差须 < tol）。

    返回：
        dict: {
            'passed': bool,               # 是否全部通过
            'max_rel_error': float,       # 所有输入中的最大相对误差
            'per_input': [ {'max_rel_error': float, 'shape': tuple}, ... ]
        }

    流程（三步）：
        1) 数组包装为 requires_grad=True 的 Node -> 前向 -> backward()
           取解析梯度；
        2) numerical_gradient 用中心差分取数值梯度；
        3) 逐元素相对误差（契约公式）比较。
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    # ---------- 1. 解析梯度：包装 Node + 前向 + 反向 ----------
    nodes = [Node(a, requires_grad=True) for a in arrays]
    out = f(nodes)
    out.backward()          # 标量输出的种子梯度默认全 1
    analytic = [(n.grad if n.grad is not None else np.zeros_like(n.data))
                for n in nodes]
    # ---------- 2. 数值梯度：中心差分（f 内部重新包装普通 Node 即可）----------
    def f_scalar(arrs):
        ns = [Node(a, requires_grad=False) for a in arrs]
        return float(f(ns).data)
    numeric = numerical_gradient(f_scalar, arrays, eps=eps)
    # ---------- 3. 相对误差逐元素比较 ----------
    per_input = []
    max_rel = 0.0
    for a, n in zip(analytic, numeric):
        rel = _rel_error(a, n)
        per_input.append({'max_rel_error': float(rel.max()),
                          'shape': tuple(a.shape)})
        max_rel = max(max_rel, float(rel.max()))
    return {'passed': bool(max_rel < tol),
            'max_rel_error': float(max_rel),
            'per_input': per_input}


# ============================================================================
#                     模型级梯度检验（全参数数值验证）
# ============================================================================
def _walk_layers(layer):
    """递归展开容器层：yield 全部叶子层（Sequential 含 layers 属性则深入）。"""
    if hasattr(layer, 'layers'):
        for sub in layer.layers:
            yield from _walk_layers(sub)
    else:
        yield layer


def check_model_gradients(model, x_np, loss_fn, y=None, eps=1e-5, tol=1e-5):
    """对完整模型做全参数数值梯度检验。

    参数：
        model: 具有 named_params()/forward/train_mode 的模型（Sequential 等）；
        x_np (np.ndarray): 输入数据；
        loss_fn: 损失函数。兼容两种用法：
            · 闭包已捕获标签：loss_fn(pred) -> 标量 Node；
            · 直接传损失对象并给 y：loss_fn(pred, y) -> 标量 Node
              （y 作为关键字参数传入本函数，属契约之上的补充设计）；
        y: 可选，标签/目标（loss_fn 需要 (pred, y) 两参时提供）；
        eps (float): 差分步长；
        tol (float): 通过阈值。

    返回：
        dict: {'passed': bool, 'max_rel_error': float,
               'per_param': {参数名: {'max_rel_error': float, 'shape': tuple}}}

    实现思路：
        解析梯度：正常前向 + backward；
        数值梯度：直接在 p.data 上做中心差分（扰动 -> 完整前向 -> 损失标量），
        全部结束后恢复原始参数并清空梯度。
        检验期间强制评估模式（避免 Dropout 等随机层破坏可重复性），结束后还原。
    """
    named = model.named_params()
    names = [n for n, _ in named]
    plist = [p for _, p in named]
    backups = [p.data.copy() for p in plist]     # 备份原始参数，最后恢复

    # ---- 保存并强制评估模式（随机层如 Dropout 会破坏数值检验）----
    saved_modes = [(l, getattr(l, 'training', None)) for l in _walk_layers(model)]
    model.train_mode(False)

    try:
        # ---------- 1. 解析梯度 ----------
        model.zero_grad()
        pred = model(Node(x_np))
        loss = loss_fn(pred) if y is None else loss_fn(pred, y)
        loss.backward()
        analytic = [(p.grad if p.grad is not None else np.zeros_like(p.data)).copy()
                    for p in plist]

        # ---------- 2. 数值梯度：逐参数逐元素中心差分 ----------
        def forward_loss():
            """以当前 p.data 完整前向一次，返回标量损失 float。"""
            pr = model(Node(x_np))
            l = loss_fn(pr) if y is None else loss_fn(pr, y)
            return float(l.data)

        per_param = {}
        max_rel = 0.0
        for k, p in enumerate(plist):
            num = np.zeros_like(p.data)
            it = np.nditer(p.data, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                orig = p.data[idx]
                p.data[idx] = orig + eps            # 直接扰动参数数据
                f_plus = forward_loss()
                p.data[idx] = orig - eps
                f_minus = forward_loss()
                p.data[idx] = orig                  # 恢复该元素
                num[idx] = (f_plus - f_minus) / (2.0 * eps)
                it.iternext()
            rel = _rel_error(analytic[k], num)
            per_param[names[k]] = {'max_rel_error': float(rel.max()),
                                   'shape': tuple(p.data.shape)}
            max_rel = max(max_rel, float(rel.max()))
            p.data[...] = backups[k]                # 该参数检验完，整体恢复
    finally:
        # ---- 无论成败都恢复：原始参数值 + 原来的训练/评估模式 ----
        for p, backup in zip(plist, backups):
            p.data = backup
        for layer, mode in saved_modes:
            if mode is not None:
                layer.train_mode(mode)
        model.zero_grad()

    return {'passed': bool(max_rel < tol),
            'max_rel_error': float(max_rel),
            'per_param': per_param}
