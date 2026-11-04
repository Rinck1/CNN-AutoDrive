# `applications/linear_regression.py` 阅读指南

> 前置：`Sequential/Linear/MSELoss/Adam/Trainer` ｜ 难度：★ ｜ 建议 20 分钟

## 1. 文件定位

最小的端到端示例：用 MiniFlows 学 `y = 3x + 2 + 噪声`。
目的是验证“前向、反向、优化器、评估”这条链路是否正确，而不是追求精度。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `make_data` (62) | 合成 200 条一维数据 |
| 2 | `build_model` (82) | `Sequential(Linear(1,1))` |
| 3 | `get_w_b` (93) | 从 `named_params` 读出 w/b |
| 4 | `train_with` (115) | 用 MBGD 或 Adam 训练 100 轮 |
| 5 | `main` (149) | 汇总打印 + 保存拟合图/损失曲线 |

## 3. 逐段讲解

### 3.1 数据

```python
x = rng.randn(n, 1)             # 列向量 (200,1)
y = 3.0*x + 2.0 + noise         # 真值 w=3, b=2；噪声 σ=1
```

噪声标准差 1.0 意味着**理论最优 MSE ≈ 1.0**；实测 0.93 已经接近极限。

### 3.2 模型

`Linear(1,1)`：W 形状 `(1,1)`，b `(1,)`。输出 `(N,1)`，与 y 形状一致。

### 3.3 训练

`train_with('MBGD')` / `train_with('Adam')`：

- 新建模型（同初始化种子）；
- `Trainer(model, MSELoss(), optimizer, task='regression')`；
- 100 轮，全量/小批量，记录 history；
- 返回最终 MSE、学到的 w/b。

### 3.4 结果解读

| 优化器 | MSE | w | b |
|---|---|---|---|
| MBGD | 0.9343 | 2.9899 | 2.0444 |
| Adam | 0.9382 | 3.0204 | 2.0394 |

两者都接近真值；MSE≈0.93 不是“差”，而是噪声方差 1.0 的下限。

## 4. 调用关系

`make_data → build_model → Trainer.train → get_w_b → evaluate`；
`plot_loss_curve` 画两种优化器对比。

## 5. 易错点

1. **W 是 `(1,1)`，不是标量**：`get_w_b` 要用 `p.data.reshape(-1)`；
2. **y 是列向量**：与模型输出 `(N,1)` 对齐，避免 `(N,)` 广播；
3. **MSE≈0.93 的解读**：要对比噪声方差，不要以为“必须接近 0”；
4. **两种优化器要同初始化**：否则对比不公平。

## 6. 自检问题

1. 为什么理论最优 MSE 是 1.0？
2. 如果 y 用 `(N,)`、模型输出 `(N,1)`，会发生什么？
3. 如何从 `named_params` 里区分 W 和 b？
