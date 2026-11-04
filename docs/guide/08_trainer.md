# `miniflows/trainer.py` 阅读指南

> 前置：layers/losses/optim/data ｜ 难度：★★ ｜ 建议 45 分钟 ｜ 核心

## 1. 文件定位

`Trainer` 把“数据分批、前向、损失、反向、更新、验证、回调”串成一个训练循环，
是应用脚本里最常用的入口。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `Trainer.__init__` (32) | 保存 model/loss/optimizer/task/seed |
| 2 | `train` | 主循环：epochs × batches，记录 history |
| 3 | `evaluate` | 验证/测试：平均损失 + 准确率/MSE |
| 4 | `predict` / `predict_classes` | 原始输出 / 分类结果 |

## 3. 逐段讲解

### 3.1 初始化

```python
Trainer(model, loss_fn, optimizer, task='classification', seed=0)
```

- `task='classification'`：`evaluate` 返回准确率；
- `task='regression'`：返回 MSE；
- `seed` 控制 DataLoader 乱序（不影响参数初始化）。

### 3.2 train 主循环

```python
for epoch in range(epochs):
    model.train_mode(True)
    for xb, yb in DataLoader(X_train, y_train, batch_size, shuffle, seed):
        pred = model(Node(xb))
        loss = loss_fn(pred, yb)
        if l2_fn: loss = loss + l2_fn()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # 验证
    model.train_mode(False)
    val_loss, val_acc = evaluate(X_val, y_val)
    history[...] append
    # 回调
    for cb in callbacks:
        if cb(state): stop = True
```

关键点：

- Batch 数据转 Node 是在 Trainer 内完成的；
- `l2_fn` 是零参函数，每个 batch 调用一次；
- 验证前必须 `train_mode(False)`，否则 Dropout 会随机；
- `callbacks` 接收 state 字典（含 epoch/val_loss/train_loss 等），返回 True 早停。

### 3.3 history

固定四个键：

```python
{'train_loss': [...], 'val_loss': [...], 'val_acc': [...], 'lr': [...]}
```

`plot_loss_curve` 依赖这些键；回归任务的 `val_acc` 实际是 MSE。

### 3.4 evaluate

- 分批前向，按样本数加权平均损失；
- 分类：多维输出 argmax；单输出按 **logit>0**（不是 0.5）；
- 回归：指标位置返回 MSE；
- 评估模式自动关闭 Dropout。

### 3.5 predict / predict_classes

- `predict`：返回原始输出（logits/回归值）；
- `predict_classes`：分类 argmax；单输出按阈值 0 判 1；回归直接返回。

## 4. 调用关系

应用脚本：`trainer.train(...)` → `trainer.evaluate(...)` → `predict`；
`EarlyStopping/LRScheduler` 通过 callbacks 接入；`plot_loss_curve` 消费 history。

## 5. 易错点

1. **单输出准确率阈值**：BCE 模型输出 logits，阈值是 0，不是 0.5；
2. **最后一个 batch**：`range(0, n, batch_size)` 会包含不足一个 batch 的尾巴；
3. **验证集/测试集**：`Trainer` 不区分，调用方不要把测试集当验证集；
4. **predict 仍建图**：没有 `no_grad`，大 batch 推理会占内存；
5. **callbacks 顺序**：多个回调按列表顺序执行，任一返回 True 即停。

## 6. 自检问题

1. 画出 train 一个 batch 的完整调用链；
2. 为什么验证前要 `train_mode(False)`？
3. 单输出分类的准确率阈值为什么是 0？
4. 回调如何实现早停？state 里应包含哪些字段？
5. 最后一个不满 batch 会不会被丢掉？
