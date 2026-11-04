# `pytorch_benchmark/train_lenet.py` 阅读指南

> 前置：`lenet5.py`、MNIST 数据格式 ｜ 难度：★★ ｜ 建议 40 分钟 ｜ 对照

## 1. 文件定位

PyTorch 对照训练脚本：

- 自己解析 MNIST idx.gz（不依赖 torchvision 的 Dataset）；
- 用与 MiniFlows 相同的类别均衡子集；
- 训练同结构 LeNet-5；
- 读取 `outputs/mnist/mnist_result.json`，打印对比表并保存报告。

## 2. 结构速览

| 符号/行号 | 作用 |
|---|---|
| `_download_one_idx` (82) | 从镜像下载单个 idx.gz |
| `_read_idx_images` (111) | 解析图像 IDX |
| `_read_idx_labels` (129) | 解析标签 IDX |
| `get_mnist_numpy` (147) | 返回 numpy 版全量数据 |
| `balanced_subset` (201) | 类别均衡子集 |
| `evaluate` (238) | 测试损失/准确率 |
| `train_model` (265) | 训练循环 |
| `plot_history` (313) | 损失曲线 |
| `print_and_save_comparison` (341) | 读 JSON → 对比表 |
| `main` (444) | 流程入口 |

## 3. 逐段讲解

### 3.1 数据

- 镜像 `https://ossci-datasets.s3.amazonaws.com/mnist/`；
- IDX 格式：魔数 + 维度 + 数据；
- 返回 `X:(N,1,28,28) float32 [0,1]`，`y:(N,) int64`。

### 3.2 子集

`balanced_subset(X, y, n)`：按类别轮流取样，逻辑必须与
`miniflows/data.py:_balanced_subset` 完全一致，否则对比不公平。

### 3.3 训练

- `LeNet5`，Adam lr=1e-3，batch=64，3 轮，CPU；
- 每轮打印 train/test loss 与准确率；
- 保存 `pytorch_result.json`。

### 3.4 对比

读取 `outputs/mnist/mnist_result.json`：

| 框架 | 准确率 | 参数量 | 耗时 |
|---|---|---|---|
| MiniFlows | 92.85% | 44,426 | 10.9s |
| PyTorch | 93.05% | 44,426 | 1.4s |

结论要写“准确率差 0.20pp，小于标准误，视为实现等价；耗时差来自 C++/MKL 与 float64”。

### 3.5 产物

- `outputs/pytorch_benchmark/pytorch损失曲线.png`；
- `pytorch_result.json`；
- `对比结果.md`。

## 4. 调用关系

`get_mnist_numpy → balanced_subset → train_model → evaluate →
print_and_save_comparison`；依赖 `lenet5.LeNet5`。

## 5. 易错点

1. 子集取样逻辑必须与自研版一致；
2. 设备必须同为 CPU，否则耗时对比无意义；
3. 读取 JSON 前要先跑 `applications/mnist.py`；
4. 0.2pp 差异不能写成“PyTorch 更好”；
5. 数据下载失败要有中文提示。

## 6. 自检问题

1. 为什么参数量必须一致？
2. 准确率差 0.2pp 如何判断是否显著？
3. 耗时差异的主要来源？
4. 如何验证两边用的是同一批样本？
