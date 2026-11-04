# `applications/mnist.py` 阅读指南

> 前置：Conv2D/Pool/CrossEntropy、`load_mnist_arrays` ｜ 难度：★★ ｜ 建议 45 分钟

## 1. 文件定位

用自研 LeNet 风格 CNN 做 MNIST 手写数字分类，是“框架能训练真实 CNN”的核心证据。
快速模式：8000 训练 / 2000 测试，3 轮，测试准确率 92.85%，约 11 秒。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `build_lenet` (81) | 定义 CNN 结构 |
| 2 | `make_time_callback` (119) | 每轮打印耗时回调 |
| 3 | `parse_args` (157) | `--full` / 快速模式 |
| 4 | `main` (171) | 数据/训练/评估/保存 |

## 3. 逐段讲解

### 3.1 网络结构（44,426 参数）

```
Conv2D(1→6, k5)   (N,1,28,28) → (N,6,24,24)
ReLU
MaxPool2D(2)                   → (N,6,12,12)
Conv2D(6→16, k5)               → (N,16,8,8)
ReLU
MaxPool2D(2)                   → (N,16,4,4)
Flatten                        → (N,256)
Linear(256→120) → ReLU
Linear(120→84)  → ReLU
Linear(84→10)
```

尺寸公式：`(28-5)//1+1=24`，`24//2=12`，`(12-5)+1=8`，`8//2=4`，`16*4*4=256`。

### 3.2 数据

`load_mnist_arrays(subset_train=8000, subset_test=2000)`：

- 类别均衡子集（按类别轮流取样）；
- X `(N,1,28,28) float32 [0,1]`，y 整型；
- 首次运行会自动下载/缓存到 `data/mnist/`。

### 3.3 训练

- Adam lr=1e-3，batch=64，3 轮；
- `Trainer` 每轮记录 loss/val_acc，`make_time_callback` 打印耗时；
- 训练结束在测试集评估一次。

### 3.4 结果与产物

| 指标 | 值 |
|---|---|
| 测试准确率 | **92.85%** |
| 测试损失 | 0.2228 |
| 参数量 | 44,426 |
| 训练耗时 | 10.9 s（CPU） |

产物：

- `outputs/mnist/训练曲线.png`；
- `outputs/mnist/预测示例.png`（8 张图，标题“真/预/(对/错)”）；
- `outputs/mnist/mnist_result.json`（供 PyTorch 对比脚本读取）。

### 3.5 `--full`

全量 60000/10000，5 轮，准确率通常更高（约 97%+），耗时数分钟。

## 4. 调用关系

`load_mnist_arrays → build_lenet → Trainer.train → evaluate →
predict → plot_images_grid → save json`；
`pytorch_benchmark/train_lenet.py` 读取同一份 json 做对比。

## 5. 易错点

1. **测试集当验证集**：`Trainer.train(..., X_val=X_test)`（`mnist.py:211`），
   指标可能偏乐观；严格协议应另切验证集；
2. **参数量**：44,426，与 PyTorch 版必须一致，否则对比不公平；
3. **预测示例标题**：不能出现“图对但标签错”的错位；
4. **JSON 字段**：framework/test_accuracy/epochs/subset/seconds/param_count/device；
5. **快速模式**：3 轮 92.85% 不是上限，别写成“MNIST 最好成绩”。

## 6. 自检问题

1. 逐层推导每层输出尺寸与参数量；
2. 为什么测试准确率能到 92.85%，而全量训练能更高？
3. `mnist_result.json` 每个字段的用途？
4. 如何改成严格的三段划分而不改变网络结构？
