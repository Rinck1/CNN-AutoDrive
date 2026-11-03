# -*- coding: utf-8 -*-
"""
miniflows/losses.py —— MiniFlows 损失函数
=========================================

【模块定位】
    实现三个核心损失（MSE / BCEWithLogits / CrossEntropy）与 L2 正则项。
    每个损失类是可调用对象：``loss = loss_fn(pred_node, target_ndarray)``
    返回标量 Node（内部直接注册高效的反向闭包，而不是拆成多个基础算子）。

【三个损失的反向公式（核心！）】
    MSELoss:
        L = mean((p - t)²)      dL/dp = 2(p - t)/N
    BCEWithLogitsLoss（数值稳定公式）:
        L = mean( max(z,0) - z·t + log(1 + e^{-|z|}) )
        dL/dz = (sigmoid(z) - t)/N
        （推导：z>0 时 d/dz[z - zt + ln(1+e^{-z})] = 1 - t - (1-σ) = σ - t；
                z<0 时 d/dz[-zt + ln(1+e^{z})] = -t + e^{z}/(1+e^{z}) = σ - t）
    CrossEntropyLoss（log-softmax + NLL）:
        L = -mean(log p[y])，其中 p = softmax(z)
        dL/dz = (softmax_onehot - onehot)/N
        （即"softmax 减去 one-hot 再除以批量大小"——分类框架最经典的梯度）

【依赖】numpy + miniflows.graph + miniflows.ops。
"""

import numpy as np

from miniflows.graph import Node, constant
from miniflows.ops import unbroadcast


# ============================================================================
#                                MSE 损失
# ============================================================================
class MSELoss:
    """均方误差损失：L = mean((pred - target)²)（回归任务标配）。

    用法：loss = MSELoss()(pred_node, target_ndarray) -> 标量 Node
    """

    def __call__(self, pred, target):
        """计算 MSE 损失。

        参数：
            pred (Node): 模型预测，任意形状；
            target (np.ndarray): 目标值（自动转 float64），可与 pred 广播。

        返回：
            Node: 标量损失。

        实现思路：
            前向直接算 mean((p-t)²)；反向闭包直接把
            dL/dp = 2(p-t)/N 写进 pred.grad（一步到位，避免拆算子）。
        """
        pred = pred if isinstance(pred, Node) else constant(pred)
        t = np.asarray(target, dtype=np.float64)
        # 与 BCEWithLogitsLoss 相同的广播陷阱：pred 为 (N,1)、target 为 (N,) 时，
        # numpy 会广播成 (N,N) 的笛卡尔组合，损失与梯度都会错但不报错。
        # 这里显式把标签补成 (N,1)，保证逐样本对应。
        if pred.data.ndim == 2 and pred.data.shape[1] == 1 and t.shape == (pred.data.shape[0],):
            t = t.reshape(-1, 1)
        # 对称情况：预测 (N,)、标签 (N,1) 时把标签压回 (N,)，避免同样的 (N,N) 广播
        elif t.ndim == 2 and t.shape[1] == 1 and pred.data.shape == (t.shape[0],):
            t = t.reshape(-1)
        diff = pred.data - t                      # 误差 (p - t)（已广播对齐）
        n = diff.size                             # 元素总数（mean 的分母）
        loss_val = np.mean(diff ** 2)             # 前向标量损失
        out = Node(loss_val, requires_grad=pred.requires_grad)
        out.parents = [pred]
        out.op_name = 'MSELoss'

        def _backward():
            if pred.requires_grad:
                # 链式：dL/dp = d/dp [mean((p-t)²)] = 2(p-t)/N，再乘种子梯度
                pred.grad += unbroadcast((2.0 * diff / n) * out.grad,
                                         pred.data.shape)
        out._backward = _backward
        return out


# ============================================================================
#                       带 logits 的二元交叉熵损失
# ============================================================================
class BCEWithLogitsLoss:
    """二元交叉熵（输入为未过 sigmoid 的 logits，内部数值稳定实现）。

    前向（数值稳定公式，避免 log(σ(z)) 在 |z| 大时溢出/下溢）：
        L = mean( max(z,0) - z·t + log(1 + e^{-|z|}) )
    反向：
        dL/dz = (sigmoid(z) - t) / N
    """

    def __call__(self, logits, target01):
        """计算 BCE 损失。

        参数：
            logits (Node): 未激活的原始输出，z:(N,1) 或 (N,) 等形状；
            target01 (np.ndarray): 目标，取值 ∈ {0,1}，形状可与 logits 广播。

        返回：
            Node: 标量损失。

        数值稳定思路：
            令 ez = e^{-|z|} ∈ (0,1]（永不上溢），则
            log(1+e^{z}) = max(z,0) + log(1+e^{-|z|})，
            于是 -[t·ln σ(z) + (1-t)·ln(1-σ(z))] = max(z,0) - z·t + log(1+e^{-|z|})。
        """
        logits = logits if isinstance(logits, Node) else constant(logits)
        z = logits.data
        t = np.asarray(target01, dtype=np.float64)
        # 关键：logits 为 (N,1)、标签为 (N,) 时，二者是"逐样本对应"关系，
        # 必须先把标签补成 (N,1)。否则 numpy 会广播成 (N,N) 的笛卡尔组合，
        # 损失值和反向梯度都会完全错误（这是一个很隐蔽的广播陷阱）。
        if z.ndim == 2 and z.shape[1] == 1 and t.shape == (z.shape[0],):
            t = t.reshape(-1, 1)
        # 广播对齐 z 与 t（允许 (N,1) 与 (N,) 之类的组合）
        zb, tb = np.broadcast_arrays(z, t)
        n = zb.size                                # 求均值的元素总数
        ez = np.exp(-np.abs(zb))                   # e^{-|z|} ∈ (0,1]：数值安全
        # 前向：max(z,0) - z*t + log(1+e^{-|z|})，等价于稳定的 -BCE
        loss_val = np.mean(np.maximum(zb, 0.0) - zb * tb + np.log1p(ez))
        out = Node(loss_val, requires_grad=logits.requires_grad)
        out.parents = [logits]
        out.op_name = 'BCEWithLogitsLoss'

        def _backward():
            if logits.requires_grad:
                # 稳定计算 sigmoid(z)：z>=0 用 1/(1+ez)；z<0 用 ez/(1+ez)
                sig = np.where(zb >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))
                # dL/dz = (sigmoid(z) - t)/N，再乘种子梯度
                dz = ((sig - tb) / n) * out.grad
                logits.grad += unbroadcast(dz, logits.data.shape)
        out._backward = _backward
        return out


# ============================================================================
#                       交叉熵损失（多分类，logits 输入）
# ============================================================================
class CrossEntropyLoss:
    """多分类交叉熵：logits (N,C) + 整型标签 (N,) -> 标量。

    内部实现 = log-softmax（减最大值稳定）+ 负对数似然（NLL）取均值：
        logp[i,c] = z[i,c] - max_j z[i,j] - log Σ_j e^{z[i,j]-max_j z[i,j]}
        L = -mean(logp[i, y_i])
    反向闭包（核心公式）：
        dL/dz = (softmax_onehot - onehot) / N
    """

    def __call__(self, logits, labels_int):
        """计算交叉熵损失。

        参数：
            logits (Node): (N,C) 未归一化得分；
            labels_int (np.ndarray): (N,) 整型类别标签（0..C-1）。

        返回：
            Node: 标量损失。
        """
        logits = logits if isinstance(logits, Node) else constant(logits)
        z = logits.data
        labels = np.asarray(labels_int).astype(np.int64).reshape(-1)
        N = z.shape[0]
        # ---------- 前向：log-softmax（减最大值防溢出）----------
        zshift = z - z.max(axis=1, keepdims=True)             # 减每行最大值
        logsumexp = np.log(np.exp(zshift).sum(axis=1, keepdims=True))  # log Σ e^{z-max}
        logprob = zshift - logsumexp                          # log-softmax (N,C)
        loss_val = -np.mean(logprob[np.arange(N), labels])    # 取标签位置的 log 概率取负均值
        out = Node(loss_val, requires_grad=logits.requires_grad)
        out.parents = [logits]
        out.op_name = 'CrossEntropyLoss'

        def _backward():
            if logits.requires_grad:
                prob = np.exp(logprob)               # softmax 概率 (N,C)
                onehot = np.zeros_like(z)
                onehot[np.arange(N), labels] = 1.0   # one-hot 编码标签
                # 经典结论：dL/dz = (softmax - onehot)/N，再乘种子梯度
                logits.grad += ((prob - onehot) / N) * out.grad
        out._backward = _backward
        return out


# ============================================================================
#                                L2 正则项
# ============================================================================
def l2_penalty(model, lam):
    """L2 权重衰减正则项：L2 = lam * Σ_p ||p||² / 2（对所有命名参数）。

    参数：
        model: 具有 named_params() 方法的模型（Layer / Sequential）；
        lam (float): 正则系数 λ。

    返回：
        Node: 标量正则项（可直接与损失相加）。

    实现思路：
        用基础算子组合（mul/sum）逐参数构造 lam/2·||p||²，梯度由通用
        反向机制自动完成：dL2/dp = lam·p（即权重衰减的来源）。
    """
    from miniflows import ops       # 局部导入，保持模块依赖清晰
    total = None
    for name, p in model.named_params():
        sq = ops.mul(p, p)                      # p ⊙ p = ||p||² 的逐元素平方
        term = ops.mul(ops.sum(sq), lam / 2.0)  # (lam/2)·||p||² 标量
        total = term if total is None else ops.add(total, term)
    if total is None:                           # 无参数模型的退化情形
        total = constant(0.0)
    total.op_name = 'L2Penalty'
    return total
