# `miniflows/utils.py` 阅读指南

> 前置：trainer ｜ 难度：★ ｜ 建议 30 分钟 ｜ 工具

## 1. 文件定位

训练辅助与持久化：随机种子、准确率/MSE、早停、学习率调度、模型保存/加载、参数量统计。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `set_seed` (29) | random + numpy 种子 |
| 2 | `accuracy` (39) | 分类准确率（argmax / 阈值） |
| 3 | `mse` (65) | 均方误差 |
| 4 | `EarlyStopping` (83) | 连续 patience 轮无改善则停 |
| 5 | `LRScheduler` (128) | step / plateau 两种衰减 |
| 6 | `save_model` (198) / `load_model` (215) | npz 按参数名存取 |
| 7 | `count_params` (242) | 参数总量 |

## 3. 逐段讲解

### 3.1 accuracy

```python
if arr.ndim > 1 and arr.shape[-1] > 1:
    pred = argmax(axis=-1)
else:
    pred = (arr.ravel() > threshold)
```

- 多分类：argmax；
- 单输出：默认阈值 0.5（概率）；如果模型输出 logits（配合 BCE），
  必须传 `threshold=0.0`。`Trainer.evaluate` 对单输出自动传 0。

### 3.2 EarlyStopping

- 记录 `best_loss`；`val_loss < best_loss - min_delta` 才算改善；
- 连续 `patience` 轮无改善返回 True；
- 只做“是否停”的判断，不负责恢复最优权重（当前版本没有 checkpoint）。

### 3.3 LRScheduler

```python
LRScheduler(opt, mode='step', step_size=10, gamma=0.5)
LRScheduler(opt, mode='plateau', patience=3, gamma=0.5)
```

- `step`：每 `step_size` 轮 `lr *= gamma`；
- `plateau`：验证损失 `patience` 轮无改善才降；
- 下限 `min_lr`；直接改 `optimizer.lr`，不建图。

### 3.4 save_model / load_model

```python
np.savez(path, **{name: p.data for name, p in model.named_params()})
```

- 参数名作为 key（如 `fc1.W`）；
- `load_model` 按名字对号入座，要求模型结构/命名完全一致；
- 只存参数，不存结构/config，加载需手工建同结构模型；
- 往返后预测应逐位一致（exp5 有断言）。

### 3.5 count_params

```python
sum(p.data.size for p in model.params())
```

只统计可训练参数；`Conv2D` 的 W/b、`Linear` 的 W/b 都会计入。

## 4. 调用关系

`Trainer` 用 accuracy；应用脚本用 EarlyStopping/LRScheduler/save_model；
exp5 验证保存加载；审核分工里把 `save/load` 作为重点检查项。

## 5. 易错点

1. **单输出阈值**：概率用 0.5，logits 用 0；
2. **EarlyStopping 不恢复权重**：训练结束后模型是最后一轮，不是最优轮；
3. **LRScheduler 共享 optimizer**：传入的是对象，直接改 `opt.lr`；
4. **参数名一致**：改层名后旧 npz 加载会报错；
5. **npz 不入库**：`.gitignore` 排除了 `*.npz`，需脚本重新生成。

## 6. 自检问题

1. `accuracy` 对 `(N,1)` logits 和 `(N,C)` logits 分别怎么处理？
2. EarlyStopping 的 `min_delta` 作用是什么？
3. step 与 plateau 调度的区别？
4. save/load 为什么要求参数名一致？如何验证往返一致？
