# `miniflows/losses.py` 阅读指南

> 前置：`graph.py`、softmax/BCE 数学 ｜ 难度：★★ ｜ 建议 45 分钟 ｜ 核心

## 1. 文件定位

四个损失/正则：`MSELoss`、`BCEWithLogitsLoss`、`CrossEntropyLoss`、`l2_penalty`。
它们把“模型输出 + 标签”变成一个标量 Node，反向时把梯度直接写回预测节点。

## 2. 结构速览

| 顺序 | 符号/行号 | 用途 |
|---|---|---|
| 1 | `MSELoss` (36) | 回归：`mean((p-t)^2)` |
| 2 | `BCEWithLogitsLoss` (85) | 二分类：logits 输入的稳定 BCE |
| 3 | `CrossEntropyLoss` (141) | 多分类：log-softmax + NLL |
| 4 | `l2_penalty` (188) | 权重 L2 正则 |

## 3. 逐段讲解

### 3.1 MSELoss

```python
diff = pred.data - t
loss = mean(diff**2)
dpred += 2 * diff / N * out.grad
```

- 若 `pred` 是 `(N,1)`、`t` 是 `(N,)`，先把 `t` reshape 成 `(N,1)`；
- 对称地，`pred` 是 `(N,)`、`t` 是 `(N,1)` 时把 `t` 压平；
- 不做这一步，numpy 会广播成 `(N,N)`，损失/梯度都错且不报错。

### 3.2 BCEWithLogitsLoss

稳定公式：

```
L = mean( max(z,0) - z*t + log(1 + e^{-|z|}) )
dz = (sigmoid(z) - t) / N
```

- `sigmoid` 用 `np.where(z>=0, 1/(1+ez), ez/(1+ez))`，避免 exp 溢出；
- 单输出模型输出的是 logits，`Trainer` 评估时阈值应为 0（sigmoid>0.5）；
- 同样处理 `(N,1)/(N,)` 的形状对齐。

### 3.3 CrossEntropyLoss

```
m = max_c z_c
logp = z - m - log(sum(exp(z-m)))
L = -mean(logp[range(N), y])
dz = (softmax(z) - onehot(y)) / N
```

- 标签是整型 `(N,)`，不是 one-hot；
- 减最大值保证 exp 不上溢；
- 反向不需要 softmax 的 L×L 雅可比，直接 `(p-onehot)/N`。

### 3.4 l2_penalty

```python
lam * sum(||W||^2) / 2
```

- **对所有 `named_params()` 生效，包括 W 和 bias**（实现里没有按名字过滤）；
- 返回标量 Node；`Trainer` 的 `l2_fn` 会在每个 batch 加进损失；
- 如果只想正则权重，需要在调用处自己构造只含 W 的列表，而不是直接传 model。

## 4. 调用关系

`Trainer.train` 每个 batch：`loss = loss_fn(pred, yb)`（可选 `+ l2_fn()`）→ `loss.backward()`。
`Trainer.evaluate` 只取 `float(loss.data)`，不反向。

## 5. 易错点

1. **BCE 输入是 logits，不是概率**：外面不要再套 sigmoid；
2. **CE 标签必须是整型类别**，传 one-hot 会报索引错误；
3. **MSE/BCE 的 `(N,1)/(N,)` 陷阱**：这是本项目修过的真 bug；
4. **l2_penalty 的系数**：`λ/2` 还是 `λ` 要与报告一致，本实现是 `λ/2`；
5. **reduction 固定为 mean**：没有 `sum/none` 选项，改 batch size 会改变损失尺度。

## 6. 自检问题

1. 写出 BCE 的稳定公式并解释 `e^{-|z|}` 为什么不会溢出；
2. CE 反向为什么是 `(softmax-onehot)/N`？
3. `(N,1)` 与 `(N,)` 广播后形状是什么？会造成什么后果？
4. L2 正则的梯度是什么？本实现为什么把 bias 也正则了？有什么影响？
