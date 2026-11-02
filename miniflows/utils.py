# -*- coding: utf-8 -*-
"""
miniflows/utils.py —— 框架增强工具模块（扩展⑤：框架优化功能）
====================================================================

本模块提供训练流程的"质量与效率"增强功能：

1. set_seed —— 统一 random + numpy 随机种子，保证实验可复现；
2. accuracy / mse —— 常用评估指标；
3. EarlyStopping —— 早停：验证损失连续 patience 轮无改善则建议提前结束训练，
   防止过拟合、节省训练时间；
4. LRScheduler —— 学习率调度：支持 阶梯衰减(step) 与 自适应衰减(plateau) 两种模式；
5. save_model / load_model —— 基于 np.savez 的轻量模型持久化（参数字典），
   支持保存后再加载的预测结果完全一致（往返一致性）；
6. count_params —— 统计模型可训练参数总量。

这些工具与 Trainer 配合使用（EarlyStopping/LRScheduler 可作为 callbacks 注入）。
"""

import os
import random

import numpy as np


# ======================================================================
# 一、随机种子与评估指标
# ======================================================================
def set_seed(seed):
    """统一设置 random 与 numpy 的全局随机种子，保证实验可复现。

    参数：
        seed (int): 随机种子。
    """
    random.seed(seed)                                       # python 内置随机源
    np.random.seed(seed)                                    # numpy 全局随机源


def accuracy(logits_or_preds, y, threshold=0.5):
    """计算分类准确率。

    自动识别输入形式：
        - 二维及以上且最后一维 > 1（如 logits (N,C)）：按 argmax 取预测类别；
        - 一维（或最后一维 == 1）：按阈值判正类。

    参数：
        logits_or_preds (np.ndarray): 模型原始输出或已算好的预测类别；
        y (np.ndarray): 真实整数标签 (N,)；
        threshold (float): 单输出时的判定阈值。默认 0.5 适用于"概率"输出；
            若模型输出的是 logits（配合 BCEWithLogitsLoss），应传 0.0
            （sigmoid(logit) > 0.5 等价于 logit > 0）。

    返回：
        float: 准确率（0~1）。
    """
    arr = np.asarray(logits_or_preds)                       # 兼容 list / Node.data
    yy = np.asarray(y).ravel()                              # 展平真实标签
    if arr.ndim > 1 and arr.shape[-1] > 1:                  # 多维输出 -> argmax 分类
        preds = arr.reshape(-1, arr.shape[-1]).argmax(axis=1)
    else:                                                   # 单输出 -> 阈值二分类
        preds = (arr.ravel() > threshold).astype(np.int64)
    return float(np.mean(preds == yy))                      # 命中率均值


def mse(pred, y):
    """计算均方误差 MSE = mean((pred - y)^2)。

    参数：
        pred (np.ndarray): 预测值；
        y (np.ndarray): 真实值。

    返回：
        float: 均方误差。
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()       # 展平方便广播对齐
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(np.mean((pred - y) ** 2))                  # 逐元素平方差的均值


# ======================================================================
# 二、早停（Early Stopping）
# ======================================================================
class EarlyStopping:
    """早停回调：验证损失连续 patience 轮无改善（改善量 < min_delta）则建议停止训练。

    用法（作为 Trainer 的 callback）：
        es = EarlyStopping(patience=5)
        trainer.train(..., callbacks=[lambda state: es(state['val_loss'])])

    属性：
        best_loss (float): 迄今为止见过的最优（最小）验证损失；
        counter (int): 连续无改善的轮数计数。
    """

    def __init__(self, patience=5, min_delta=1e-4):
        """初始化早停器。

        参数：
            patience (int): 容忍连续多少轮无改善；
            min_delta (float): 视为"改善"所需的最小损失下降量。
        """
        self.patience = int(patience)                       # 容忍轮数
        self.min_delta = float(min_delta)                   # 改善阈值
        self.best_loss = float("inf")                       # 初始最优损失为正无穷
        self.counter = 0                                    # 无改善计数器清零

    def __call__(self, val_loss):
        """每轮验证后调用：更新计数器并判断是否应提前停止。

        参数：
            val_loss (float): 本轮验证损失。

        返回：
            bool: True 表示连续 patience 轮无改善，建议停止训练。
        """
        val_loss = float(val_loss)
        if val_loss < self.best_loss - self.min_delta:      # 出现"足够大"的改善
            self.best_loss = val_loss                       # 刷新最优损失
            self.counter = 0                                # 计数器归零
            return False                                    # 不停止
        self.counter += 1                                   # 无改善：计数 +1
        return self.counter >= self.patience                # 达到容忍上限 -> 建议停止


# ======================================================================
# 三、学习率调度器（LRScheduler）
# ======================================================================
class LRScheduler:
    """学习率调度器：按策略衰减 optimizer.lr，帮助训练后期稳定收敛。

    两种模式：
        - 'step'：每 step_size 轮学习率乘以 gamma（阶梯衰减，确定性调度）；
        - 'plateau'：验证损失连续 patience 轮无改善时学习率乘以 gamma
          （自适应调度，val_loss 无变化时不降）。

    参数：
        optimizer (Optimizer): 任意带 .lr 属性的优化器；
        mode (str): 'step' 或 'plateau'；
        step_size (int): step 模式的衰减周期（轮）；
        gamma (float): 每次衰减的乘法因子（<1）；
        patience (int): plateau 模式的容忍轮数；
        min_lr (float): 学习率下限，衰减到该值后不再降低。
    """

    def __init__(self, optimizer, mode="step", step_size=10, gamma=0.5,
                 patience=3, min_lr=1e-6):
        """初始化调度器并记录初始学习率。"""
        self.optimizer = optimizer                          # 目标优化器（需有 .lr 属性）
        self.mode = mode                                    # 调度模式
        self.base_lr = float(optimizer.lr)                  # 记录初始学习率（step 模式基准）
        self.step_size = int(step_size)                     # 阶梯周期
        self.gamma = float(gamma)                           # 衰减因子
        self.patience = int(patience)                       # plateau 容忍轮数
        self.min_lr = float(min_lr)                         # 学习率下限
        self.best_loss = float("inf")                       # plateau 模式的历史最优损失
        self.counter = 0                                    # plateau 无改善计数器
        self._epoch = 0                                     # 内部轮次计数（epoch 未传时自增）

    def step(self, epoch=None, val_loss=None):
        """推进一轮调度：计算并写回新学习率。

        参数：
            epoch (int | None): 当前轮次（0 起）；None 则内部自增；
            val_loss (float | None): 本轮验证损失（plateau 模式必需）。

        返回：
            float: 更新后的学习率。
        """
        if self.mode == "step":
            # ---------- 阶梯衰减：lr = base_lr * gamma^(epoch // step_size) ----------
            if epoch is None:                               # 未显式传轮次则内部计数
                self._epoch += 1
                epoch = self._epoch
            new_lr = self.base_lr * (self.gamma ** (epoch // self.step_size))
            new_lr = max(new_lr, self.min_lr)               # 不低于下限
        else:
            # ---------- 自适应衰减：验证损失 patience 轮无改善则 lr *= gamma ----------
            if val_loss is None:                            # 没有验证损失就不做任何事
                return self.optimizer.lr
            if val_loss < self.best_loss:                   # 有改善：刷新最优并清零计数
                self.best_loss = float(val_loss)
                self.counter = 0
            else:                                           # 无改善：计数 +1
                self.counter += 1
                if self.counter >= self.patience:           # 达到容忍上限 -> 衰减一次
                    new_lr = max(self.optimizer.lr * self.gamma, self.min_lr)
                    self.optimizer.lr = new_lr              # 写回优化器
                    self.counter = 0                        # 重置计数，等待下一个平台期
                    return new_lr
            return self.optimizer.lr                        # 未触发衰减：保持原学习率
        self.optimizer.lr = new_lr                          # step 模式：写回新学习率
        return new_lr


# ======================================================================
# 四、模型保存 / 加载（np.savez 轻量持久化）
# ======================================================================
def save_model(model, path):
    """把模型全部可训练参数保存为 .npz 文件（键 = 参数名，值 = 参数数组）。

    实现思路：
        调用 model.named_params() 得到 [(名字, Node), ...]，
        np.savez 直接以参数名为键保存数组。

    参数：
        model (Sequential | Layer): 任意实现 named_params() 的模型；
        path (str): 保存路径（自动创建父目录，建议 .npz 后缀）。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)  # 保证目录存在
    params = {name: np.asarray(p.data) for name, p in model.named_params()}    # 名字 -> 数组
    np.savez(path, **params)                                # 压缩打包保存
    return path


def load_model(model, path):
    """从 .npz 文件原地加载参数到模型（按参数名一一对应 copy 到 p.data）。

    实现思路：
        np.load 读出全部数组；遍历 model.named_params()，
        名字匹配的参数用 np.copyto 原地覆盖（形状必须一致才安全）。

    参数：
        model (Sequential | Layer): 待加载的目标模型（结构需与保存时一致）；
        path (str): .npz 文件路径。

    返回：
        model: 原地更新后的模型对象（便于链式使用）。
    """
    with np.load(path) as npz:                              # with 保证文件句柄关闭
        loaded = {k: npz[k] for k in npz.files}             # 先复制出来，避免句柄关闭后失效
    for name, p in model.named_params():                    # 按名字逐个对号入座
        if name not in loaded:                              # 缺参数：提示但不中断
            print(f"[utils] 警告: 保存文件中缺少参数 {name}")
            continue
        arr = np.asarray(loaded[name], dtype=np.float64)    # 统一 float64（与 Node 内部一致）
        if p.data.shape != arr.shape:                       # 形状不匹配：报错更安全
            raise ValueError(f"参数 {name} 形状不匹配: 模型 {p.data.shape} vs 文件 {arr.shape}")
        np.copyto(p.data, arr)                              # 原地拷贝，保持计算图引用不变
    return model


def count_params(model):
    """统计模型可训练参数总量。

    参数：
        model (Sequential | Layer): 任意实现 named_params() 的模型。

    返回：
        int: 所有参数的元素总数。
    """
    return int(sum(int(np.asarray(p.data).size) for _, p in model.named_params()))
