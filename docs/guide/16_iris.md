# `applications/iris.py` 阅读指南

> 前置：多分类、CrossEntropy、标准化 ｜ 难度：★★ ｜ 建议 30 分钟

## 1. 文件定位

鸢尾花三分类，对比两个模型：

1. 线性 Softmax 回归 `Linear(4,3)`；
2. 两层 MLP `Linear(4,16)+ReLU+Linear(16,3)`。

重点不是刷分，而是演示“线性 vs 非线性”的差距，以及正确的预处理顺序。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `load_and_preprocess` (67) | 加载、分层划分、标准化 |
| 2 | `build_softmax_regression` (95) | 线性模型 |
| 3 | `build_mlp` (108) | 两层 MLP |
| 4 | `train_one` (121) | 训练 + 评估 |
| 5 | `main` (148) | 两个模型对比 + 出图 |

## 3. 逐段讲解

### 3.1 预处理顺序（关键）

```python
X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_size=0.3,
                                          stratify=y, seed=42)
X_tr_s, scaler = standard_scale(X_tr)   # 只在训练集 fit
X_te_s = apply_scale(X_te, scaler)      # 测试集只 transform
```

- 先划分，再标准化，避免测试集统计量泄漏；
- 分层抽样保证 3 类比例一致（105/45）。

### 3.2 两个模型

```python
Linear(4,3)                                  # 15 参数
Linear(4,16) → ReLU → Linear(16,3)           # 131 参数
```

### 3.3 训练

Adam lr=1e-3，100 轮，batch=16，CrossEntropyLoss；
训练集训练，测试集作为验证/评估（这里有一个方法学问题，见第 5 节）。

### 3.4 结果

| 模型 | 测试准确率 |
|---|---|
| 线性 Softmax | 77.78% |
| 两层 MLP | **93.33%** |

MLP 的 ReLU 隐层能拟合弯曲的类间边界；鸢尾花本身近似线性可分，
所以线性模型也不算太差。

## 4. 调用关系

`load_iris → train_test_split → standard_scale/apply_scale →
build_* → Trainer.train → evaluate → plot`。

## 5. 方法学注意

`train_one` 把测试集同时作为 `X_val` 传给 `Trainer.train`（`iris.py:142`）。
这会让训练曲线监控到测试集，最终指标可能偏乐观。
严格做法：从训练集再切 10%~15% 作验证集，测试集只在最后评估一次。
本项目在 `donkeycar_eval.py` 里采用了严格协议，可以作为参照。

## 6. 易错点

1. **标准化泄漏**：绝不能在测试集上重新 fit；
2. **分层**：`stratify=y` 不能漏；
3. **测试集当验证集**：指标偏乐观；
4. **MLP 参数量**：131，不是“越大越好”；
5. **交叉熵标签**：整型类别，不是 one-hot。

## 7. 自检问题

1. 为什么先划分再标准化？
2. 分层抽样解决了什么问题？
3. MLP 比线性模型高 15 个百分点的原因？
4. 如何把当前评估改成严格的“训练/验证/测试”三段？
