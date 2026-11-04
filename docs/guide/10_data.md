# `miniflows/data.py` 阅读指南

> 前置：numpy、sklearn/torchvision 可选 ｜ 难度：★★ ｜ 建议 60 分钟 ｜ 工具

## 1. 文件定位

数据预处理与加载：缩放、one-hot、分层划分、合成数据、DataLoader、
鸢尾花三级降级加载、MNIST 三级降级加载、类别均衡子集、平移增强。

## 2. 结构速览

| 区块 | 行号 | 内容 |
|---|---|---|
| 缩放 | 45~124 | `standard_scale`、`minmax_scale`、`apply_scale` |
| 编码/划分 | 125~210 | `one_hot`、`train_test_split`（支持 stratify） |
| 合成数据 | 211~253 | `make_blobs` |
| DataLoader | 254~297 | 乱序分批 |
| 鸢尾花 | 298~383 | sklearn → 本地缓存 → 合成数据 |
| MNIST | 384~536 | torchvision → 镜像下载 → 本地 idx.gz 解析 |
| 子集/增强 | 537~588 | 类别均衡子集、随机平移 |

## 3. 逐段讲解

### 3.1 缩放（防数据泄漏）

```python
X_tr_s, scaler = standard_scale(X_tr)
X_te_s = apply_scale(X_te, scaler)
```

- `standard_scale` 返回 `(X_scaled, {'mean','std'})`；
- `apply_scale` 只做变换，绝不在测试集上重新 fit；
- `minmax_scale` 类似，返回 `{'min','max'}`；
- 常量特征 std=0 时要有保护（避免除零）。

### 3.2 划分

```python
train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, seed=0)
```

- `stratify=y`：按类别比例分层抽样，保持训练/测试分布一致；
- 返回 `(X_tr, y_tr, X_te, y_te)`；
- 必须保证训练/测试索引不重叠（测试里用集合验证）。

### 3.3 DataLoader

```python
for xb, yb in DataLoader(X, y, batch_size=32, shuffle=True, seed=0):
    ...
```

- 每轮 `permutation` 乱序；
- `seed` 固定后顺序可复现；
- 最后一个不满 batch 也会返回（不丢样本）。

### 3.4 make_blobs

两团高斯：`centers` 个中心，`center_box` 控制中心范围，`std` 控制团内散度。
逻辑回归用它生成近似线性可分的数据。

### 3.5 鸢尾花三级降级

1. `sklearn.datasets.load_iris`；
2. 本地缓存 `data/iris_cache.csv`；
3. 合成高斯团簇（打印警告）。

返回 `(X:(150,4), y:(150,), class_names)`。

### 3.6 MNIST 三级降级

1. `torchvision.datasets.MNIST`（首选）；
2. urllib 从镜像 `https://ossci-datasets.s3.amazonaws.com/mnist/` 下载 4 个 idx.gz；
3. 本地 `data/mnist/` 缓存解析。

`load_mnist` 返回 `(X:(N,1,28,28) float32 [0,1], y:(N,) int64)`。
`_parse_idx_images/labels` 手动解析 IDX 格式（魔数 + 维度 + 数据）。

### 3.7 类别均衡子集

`_balanced_subset(X, y, n)`：按类别轮流取样，保证子集类别均衡；
`load_mnist_arrays(subset_train=8000, subset_test=2000)` 用它做快速模式。

### 3.8 平移增强

`random_shift_images(X, max_shift=2, seed=0)`：对每张图做整数像素的循环平移，
模拟车辆横向偏移；注意返回 dtype/范围与输入一致。

## 4. 调用关系

应用脚本：`load_iris/load_mnist_arrays`、`standard_scale/apply_scale`、
`train_test_split`、`DataLoader`；exp4 用缩放/one-hot/分层/平移做演示。

## 5. 易错点

1. **数据泄漏**：缩放统计量只能用训练集；
2. **分层抽样**：`stratify` 在某一类样本过少时可能报错；
3. **MNIST 缓存路径**：默认 `data/mnist/`，首次运行自动下载约 12MB；
4. **dtype/范围**：MNIST 是 [0,1] float32，Node 内部会转 float64；
5. **平移增强**：循环平移会让边界内容绕回，语义上要说明；
6. **子集均衡**：不是简单取前 N 个，而是按类别轮流取。

## 6. 自检问题

1. 为什么 `apply_scale` 不能重新 fit？
2. 分层划分如何保证测试集类别比例？
3. DataLoader 一个 epoch 是否覆盖全部样本？
4. MNIST 缓存损坏时会怎样？三级降级分别在哪？
5. 类别均衡子集与随机子集的区别？
