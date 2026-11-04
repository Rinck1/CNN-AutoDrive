# `applications/logistic_regression.py` 阅读指南

> 前置：BCEWithLogitsLoss、二分类 ｜ 难度：★ ｜ 建议 25 分钟

## 1. 文件定位

二分类示例：两团二维高斯 → `Linear(2,1)` + BCEWithLogits → 决策边界可视化。
用于验证“单层线性分类器 + 稳定 BCE + Adam”是否正确。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `build_model` (60) | `Sequential(Linear(2,1))` |
| 2 | `plot_decision_boundary` (71) | 网格预测 + 等高线 |
| 3 | `main` (120) | 数据/训练/评估/出图 |

## 3. 逐段讲解

### 3.1 数据

```python
X, y = make_blobs(n_samples=300, centers=2,
                  center_box=(-5,5), std=0.7,
                  n_features=2, seed=42)
```

- 两团中心距约 5，团内 std=0.7 → 近似线性可分；
- 总数 300，分层划分 210 训练 / 90 测试。

### 3.2 模型与损失

`Linear(2,1)` 输出 logits（不过 sigmoid），`BCEWithLogitsLoss` 内部做稳定 sigmoid + 交叉熵。
**不要在模型里再加 Sigmoid**，否则损失内部会重复。

### 3.3 训练

Adam lr=0.05，200 轮，batch=32；训练集/测试集都记录损失。
最终训练损失 0.000441、测试损失 0.001236、测试准确率 100%。

### 3.4 决策边界

在特征平面取 300×300 网格：

1. `trainer.predict(grid)` 得 logits；
2. sigmoid 得 P(y=1)；
3. `contourf` 填两类区域，`contour(level=0.5)` 画边界；
4. 叠加真实散点。

边界方程由 W（形状 `(2,1)`）与 b 读出：

```
4.1002*x1 - 2.9606*x2 + 4.7009 = 0
```

### 3.5 指标解读

100% 是因为数据本身几乎线性可分，不代表模型能力强；
若想更有区分度，可缩小中心距/增大 std。

## 4. 调用关系

`make_blobs → train_test_split → Trainer.train → evaluate → plot_decision_boundary`；
`plot_loss_curve` 画训练/验证曲线。

## 5. 易错点

1. **BCE 输入是 logits**：模型里不要加 Sigmoid；
2. **W 形状 `(2,1)`**：读边界时按列向量取 `w[0], w[1]`，不是 `p.data[0,1]`；
3. **准确率阈值**：单输出 logits 用 `>0`，不是 `>0.5`；
4. **数据太容易**：100% 不是“模型很强”，报告里要说明。

## 6. 自检问题

1. 为什么 BCEWithLogits 比 Sigmoid+BCE 数值稳定？
2. 决策边界为什么是 `w·x+b=0`？
3. 如果把中心距改成 2、std 改成 1，准确率会怎样？
