# -*- coding: utf-8 -*-
"""
miniflows/trainer.py —— 训练器模块（Keras 风格 fit/predict）
================================================================

本模块把"数据批次迭代 + 前向 + 损失 + 反向 + 参数更新 + 验证评估 +
回调（早停/学习率调度）"串成一条完整的训练流水线，对外暴露 Keras 风格的
``Trainer(...).fit(...)`` 用法。

训练一个 epoch 的完整流程：
    1. model.train_mode(True)   —— 打开 Dropout 等训练期行为；
    2. DataLoader 按 batch 迭代 (xb, yb)；
    3. xb/yb 包装成 Node（数据进计算图）；
    4. pred = model(x)          —— 前向传播；
    5. loss = loss_fn(pred, yb) —— 计算损失；若提供 l2_fn 则叠加正则项；
    6. loss.backward()          —— 反向传播求梯度；
    7. optimizer.step()         —— 按梯度更新参数；
    8. optimizer.zero_grad()    —— 清空梯度，准备下一批。
验证前 model.train_mode(False)，保证 Dropout 关闭、结果可复现。

task='classification' 时 evaluate 返回 (平均损失, 准确率)；
task='regression'     时 evaluate 返回 (平均损失, MSE)（准确率位置放 MSE）。
"""

import numpy as np

from miniflows.graph import Node
from miniflows.data import DataLoader
from miniflows.utils import accuracy


class Trainer:
    """通用训练器：封装小批量训练 / 验证 / 预测的完整循环。

    参数：
        model (Sequential | Layer): 待训练模型（需实现 forward/__call__、
            train_mode(bool)、zero_grad()）；
        loss_fn (callable): 损失对象或函数，调用形式 loss_fn(pred: Node, y: np.ndarray) -> Node；
        optimizer (Optimizer): 优化器（需有 lr / step() / zero_grad()）；
        task (str): 'classification'（分类，评估准确率）或 'regression'（回归，评估 MSE）；
        seed (int): 训练用随机种子（控制 DataLoader 打乱顺序，保证可复现）。

    注意 l2_fn 的约定：train(l2_fn=...) 传入**零参可调用**，返回标量正则 Node，
    推荐写法 ``l2_fn=lambda: l2_penalty(model, 1e-4)``。
    """

    def __init__(self, model, loss_fn, optimizer, task="classification", seed=0):
        """保存模型/损失/优化器引用并初始化任务类型与随机种子。"""
        self.model = model                                  # 待训练模型
        self.loss_fn = loss_fn                              # 损失函数（对象或函数均可）
        self.optimizer = optimizer                          # 优化器
        assert task in ("classification", "regression"), f"未知任务类型: {task}"
        self.task = task                                    # 任务类型（决定 evaluate 行为）
        self.seed = int(seed)                               # 随机种子（DataLoader 用）
        # 训练历史：每次 fit 会重置
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=10, batch_size=32, shuffle=True,
              callbacks=None, l2_fn=None, verbose=True):
        """训练模型并返回历史记录（Keras 风格）。

        参数：
            X_train (np.ndarray): 训练特征；
            y_train (np.ndarray): 训练标签（分类传整数标签，回归传实数目标）；
            X_val / y_val: 验证集（可 None；给出则每轮验证并记录曲线）；
            epochs (int): 训练轮数；
            batch_size (int): 小批量大小；
            shuffle (bool): 是否每轮打乱训练集；
            callbacks (list[fn(state: dict) -> bool] | None): 回调列表，
                state 含 epoch/train_loss/val_loss/val_acc/lr 等键，
                任一回调返回 True 则提前停止训练（早停/LR调度均以此注入）；
            l2_fn (callable | None): 零参可调用，返回标量 L2 正则 Node，
                例如 ``lambda: l2_penalty(model, 1e-4)``；
            verbose (bool): 是否打印每轮日志。

        返回：
            dict: history = {'train_loss', 'val_loss', 'val_acc', 'lr'} 各为逐轮列表。
        """
        # 每次调用 fit 重置历史，支持同一 Trainer 多次实验互不污染
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
        loader = DataLoader(X_train, y_train, batch_size=batch_size,
                            shuffle=shuffle, seed=self.seed)   # 训练批迭代器（种子可复现）
        n_batches = max(len(loader), 1)                     # 每轮批数（打印进度用）

        for epoch in range(1, epochs + 1):                  # 逐轮训练
            # ---------- 1. 训练阶段 ----------
            self.model.train_mode(True)                     # 开启 Dropout 等训练期行为
            batch_losses = []                               # 收集本轮各批损失
            for xb, yb in loader:                           # 逐批迭代
                x = Node(xb)                                # 特征批 -> Node（进计算图）
                pred = self.model(x)                        # 前向传播
                loss = self.loss_fn(pred, yb)               # 计算损失（标量 Node）
                if l2_fn is not None:                       # 叠加 L2 正则项（若有）
                    loss = loss + l2_fn()
                loss.backward()                             # 反向传播：从损失节点求全图梯度
                self.optimizer.step()                       # 用梯度更新参数
                self.optimizer.zero_grad()                  # 清空梯度，避免跨批累加
                batch_losses.append(float(loss.data))       # 记录本批标量损失
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            self.history["train_loss"].append(train_loss)   # 记录训练损失

            # ---------- 2. 验证阶段（可选） ----------
            val_loss, val_acc = None, None
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)  # evaluate 内部会关闭 train_mode
                self.history["val_loss"].append(val_loss)   # 记录验证损失
                self.history["val_acc"].append(val_acc)     # 分类=准确率 / 回归=MSE

            # ---------- 3. 记录学习率（调度器可在回调里改 lr） ----------
            self.history["lr"].append(float(getattr(self.optimizer, "lr", float("nan"))))

            # ---------- 4. 组装回调 state 并执行回调 ----------
            state = {"epoch": epoch, "epochs": epochs,      # 本轮完整状态（回调只读）
                     "train_loss": train_loss,
                     "val_loss": val_loss, "val_acc": val_acc,
                     "lr": self.history["lr"][-1],
                     "model": self.model, "trainer": self}
            stop = False                                    # 是否提前停止
            for cb in (callbacks or []):                    # 任一回调返回 True 即停止
                if cb(state):
                    stop = True

            # ---------- 5. 打印日志 ----------
            if verbose:
                msg = f"[Trainer] 轮次 {epoch:3d}/{epochs} | 训练损失 {train_loss:.4f}"
                if val_loss is not None:                    # 有验证集则补充验证指标
                    msg += f" | 验证损失 {val_loss:.4f}"
                    if self.task == "classification":
                        msg += f" | 验证准确率 {val_acc:.4f}"
                print(msg)

            if stop:                                        # 回调要求提前停止
                if verbose:
                    print(f"[Trainer] 第 {epoch} 轮被回调提前终止")
                break

        return self.history                                 # 返回完整训练历史供画图

    # ------------------------------------------------------------------
    # 评估 / 预测
    # ------------------------------------------------------------------
    def evaluate(self, X, y, batch_size=256):
        """在给定数据上评估：返回 (平均损失, 指标)。

        指标含义由任务类型决定：
            classification —— 指标 = 准确率（多维输出 argmax；单维输出按 >0.5 判 1）；
            regression     —— 指标 = MSE（放在"准确率"位置，便于统一接收两个返回值）。

        参数：
            X (np.ndarray): 特征；
            y (np.ndarray): 标签/目标；
            batch_size (int): 评估批大小（大批量一次前向即可）。

        返回：
            (avg_loss, metric) 元组。
        """
        self.model.train_mode(False)                        # 关闭 Dropout：评估恒等
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)                                          # 样本总数
        total_loss, all_preds = 0.0, []                     # 累计损失 / 收集预测
        for start in range(0, n, batch_size):               # 分批前向（控内存）
            xb = X[start:start + batch_size]                # 特征切片
            yb = y[start:start + batch_size]                # 标签切片
            pred = self.model(Node(xb))                     # 前向（评估模式，不建反向）
            loss = self.loss_fn(pred, yb)                   # 本批损失
            total_loss += float(loss.data) * len(xb)        # 按批样本数加权累计
            all_preds.append(np.asarray(pred.data))         # 收集预测输出
        avg_loss = total_loss / max(n, 1)                   # 平均损失
        preds = np.concatenate(all_preds, axis=0)           # 拼接全部预测
        if self.task == "regression":
            metric = float(np.mean((preds.ravel() - y.ravel()) ** 2))  # 回归：MSE
        else:
            # 分类：多维输出走 argmax；单输出时模型给的是 logits（配合
            # BCEWithLogitsLoss），因此阈值取 0 而不是 0.5
            if preds.ndim > 1 and preds.shape[-1] == 1:
                metric = accuracy(preds, y, threshold=0.0)
            else:
                metric = accuracy(preds, y)
        return float(avg_loss), float(metric)

    def predict(self, X, batch_size=256):
        """批量预测，返回模型原始输出（logits 或回归值），不做 argmax。

        参数：
            X (np.ndarray): 特征；
            batch_size (int): 批大小。

        返回：
            np.ndarray: 原始输出（形状随模型而定，如 (N,C) logits）。
        """
        self.model.train_mode(False)                        # 评估模式（Dropout 关闭）
        X = np.asarray(X)
        outs = []                                           # 收集各批输出
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            pred = self.model(Node(xb))                     # 纯前向
            outs.append(np.asarray(pred.data))
        return np.concatenate(outs, axis=0)                 # 拼接为完整输出

    def predict_classes(self, X, batch_size=256):
        """预测类别：多维输出取 argmax；回归任务等价于 predict。

        参数：
            X (np.ndarray): 特征；
            batch_size (int): 批大小。

        返回：
            np.ndarray: 预测类别 (N,) 整数（回归时为原始预测值）。
        """
        preds = self.predict(X, batch_size=batch_size)      # 先取原始输出
        if self.task == "regression":
            return preds                                    # 回归：直接返回预测值
        if preds.ndim > 1 and preds.shape[-1] > 1:          # 多维输出 -> argmax 类别
            return preds.argmax(axis=1)
        return (preds.ravel() > 0.5).astype(np.int64)       # 单维输出 -> 0.5 阈值二分类
