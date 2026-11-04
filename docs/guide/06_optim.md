# `miniflows/optim.py` 阅读指南

> 前置：梯度下降、指数滑动平均 ｜ 难度：★★ ｜ 建议 30 分钟 ｜ 核心

## 1. 文件定位

两个优化器：`MBGD`（小批量梯度下降，可选动量）与 `Adam`。
共同约定：`params` 是 Node 列表；`step()` 直接改 `p.data`，**不建图**；
`zero_grad()` 把每个参数的 `grad` 置 None。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `Optimizer` (31) | 基类：保存参数、lr、zero_grad |
| 2 | `MBGD` (60) | `θ -= lr*g`；momentum>0 时 `v=μv+g; θ-=lr*v` |
| 3 | `Adam` (97) | 一阶/二阶矩 + 偏差修正 |

## 3. 逐段讲解

### 3.1 基类

```python
class Optimizer:
    def __init__(self, params, lr): self.params=list(params); self.lr=lr
    def zero_grad(self):
        for p in self.params: p.grad = None
    def step(self): raise NotImplementedError
```

`zero_grad` 置 None 而不是置零数组，省内存，也让“是否已反向”可判断。

### 3.2 MBGD

```python
if momentum == 0:
    p.data -= lr * p.grad
else:
    v = momentum * v + p.grad
    p.data -= lr * v
```

- 动量缓存的形状与参数一致；
- 学习率可被 `LRScheduler` 直接改 `opt.lr`；
- 不做梯度裁剪，爆炸需要外部处理（见 exp3）。

### 3.3 Adam

```python
t += 1
m = β1*m + (1-β1)*g
v = β2*v + (1-β2)*g²
m_hat = m / (1-β1^t)
v_hat = v / (1-β2^t)
p.data -= lr * m_hat / (sqrt(v_hat) + eps)
```

- `m/v` 与参数同形，初始化为 0；
- 偏差修正让第一步更新量约为 `lr*sign(g)`；
- `eps` 加在分母，防止除零；
- 所有计算在 numpy 里做，不进入计算图。

## 4. 调用关系

`Trainer` 每个 batch：`loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`。
`LRScheduler` 通过修改 `optimizer.lr` 生效。

## 5. 易错点

1. **必须先 backward 再 step**：`p.grad` 为 None 时 step 会报错/跳过；
2. **step 后必须 zero_grad**：否则梯度跨 batch 累加；
3. **不要在 step 里用算子**：直接改 `p.data`，否则计算图无限增长；
4. **Adam 的 t 从 1 开始**：偏差修正分母 `1-β^t`；
5. **学习率调度器与优化器共享对象**：直接改 `opt.lr` 即可。

## 6. 自检问题

1. 写出 Adam 的完整更新式，并解释偏差修正的作用；
2. 为什么优化器更新不能建图？
3. 动量法与 Adam 的一阶矩有什么区别？
4. `zero_grad` 置 None 与置 0 的差异？
