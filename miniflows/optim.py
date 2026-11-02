# -*- coding: utf-8 -*-
"""
miniflows/optim.py —— MiniFlows 优化器（MBGD / Adam）
====================================================

【模块定位】
    优化器负责"用梯度更新参数"。与计算图解耦：``step()`` 直接原地修改
    ``p.data``，完全不创建 Node / 不建图（更新本身不需要参与自动微分）。

【更新公式】
    MBGD（小批量梯度下降，含可选动量）：
        无动量:   p ← p - lr·g
        有动量:   v ← μ·v + g ；  p ← p - lr·v
        （动量把历史梯度指数加权平均进来，抑制震荡、加速穿越平坦区）
    Adam（自适应矩估计，含偏差修正）：
        m ← β1·m + (1-β1)·g          （一阶矩：梯度均值）
        v ← β2·v + (1-β2)·g²         （二阶矩：梯度未中心化方差）
        m̂ = m/(1-β1^t) ；v̂ = v/(1-β2^t)   （偏差修正：抵消 t 小时 m/v 偏小）
        p ← p - lr·m̂/(√v̂ + ε)
    （t 为全局步数计数；β1=0.9, β2=0.999, ε=1e-8 为论文默认值）

【依赖】仅 miniflows.graph（用到参数是 Node 这一事实）。
"""

import numpy as np


# ============================================================================
#                                优化器基类
# ============================================================================
class Optimizer:
    """优化器基类：管理参数列表，规定 zero_grad/step 接口。

    属性：
        params (list[Node]): 待优化的参数节点列表；
        lr (float): 学习率（公开属性，学习率调度器会直接修改它）。
    """

    def __init__(self, params, lr):
        """参数：
            params (list[Node]): 参数节点列表（通常来自 model.params()）；
            lr (float): 初始学习率。
        """
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        """把所有参数的梯度清空（置 None），准备下一轮累积。"""
        for p in self.params:
            p.grad = None

    def step(self):
        """执行一步参数更新。子类必须实现（直接改 p.data，不建图）。"""
        raise NotImplementedError('优化器子类必须实现 step()')


# ============================================================================
#                    MBGD：小批量梯度下降（动量可选）
# ============================================================================
class MBGD(Optimizer):
    """小批量梯度下降（Mini-Batch Gradient Descent），动量可选。

    用法：opt = MBGD(model.params(), lr=0.05, momentum=0.9)
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        """参数：
            params (list[Node]): 参数节点列表；
            lr (float): 学习率；
            momentum (float): 动量系数 μ，>0 时启用动量（常用 0.9）。
        """
        super().__init__(params, lr)
        self.momentum = momentum
        # 每个参数配一个同形速度缓冲 v（与参数一一对应、顺序一致）
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """一步更新：无动量 p -= lr·g；有动量 v = μv + g, p -= lr·v。"""
        for p, v in zip(self.params, self.velocities):
            if p.grad is None:          # 该参数本轮没参与前向 -> 跳过
                continue
            g = p.grad
            if self.momentum > 0.0:
                # 原地更新速度缓冲：v ← μ·v + g（指数加权的历史梯度累积）
                v *= self.momentum
                v += g
                # 参数沿"平滑后的速度"方向下降（直接改 data，不建图）
                p.data -= self.lr * v
            else:
                # 纯小批量梯度下降：p ← p - lr·g
                p.data -= self.lr * g


# ============================================================================
#                       Adam：自适应矩估计（偏差修正）
# ============================================================================
class Adam(Optimizer):
    """Adam 优化器（Kingma & Ba, 2015），含偏差修正。

    用法：opt = Adam(model.params(), lr=1e-3)
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """参数：
            params (list[Node]): 参数节点列表；
            lr (float): 学习率（步长上界）；
            beta1 (float): 一阶矩衰减率（默认 0.9）；
            beta2 (float): 二阶矩衰减率（默认 0.999）；
            eps (float): 数值稳定小量（防除零）。
        """
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0                                            # 全局步数计数
        # 每个参数配一对同形矩缓冲 m（一阶）与 v（二阶）
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """一步 Adam 更新（公式见模块头注释；直接改 p.data，不建图）。"""
        self.t += 1                     # 全局步数：用于偏差修正的分母
        b1, b2 = self.beta1, self.beta2
        # 预先算好偏差修正系数：1/(1-β^t)（t=1 时为 10/9.99 等，逐步趋近 1）
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:          # 本轮未参与的参数跳过
                continue
            g = p.grad
            # 一阶矩 m ← β1·m + (1-β1)·g（梯度的指数加权均值）
            m *= b1
            m += (1.0 - b1) * g
            # 二阶矩 v ← β2·v + (1-β2)·g²（梯度平方的指数加权均值）
            v *= b2
            v += (1.0 - b2) * (g * g)
            # 偏差修正：m̂ = m/(1-β1^t)，v̂ = v/(1-β2^t)
            # （否则前几步 m/v 接近 0，更新幅度会被严重低估）
            m_hat = m / bc1
            v_hat = v / bc2
            # 自适应步长更新：大梯度方向走小步、小梯度方向走大步
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
