# `pytorch_benchmark/compare_resources.py` 阅读指南

> 前置：`profiler.py`、PyTorch 基础 ｜ 难度：★★ ｜ 建议 35 分钟

## 1. 文件定位

课程要求“与其他框架比较训练时间、资源占用、FLOPS”。本脚本在**完全相同条件**下
对比 MiniFlows 与 PyTorch：

| 条件 | 设置 |
|---|---|
| 模型 | LeNet-5（参数量均 44,426） |
| 数据 | 同一份 MNIST 类别均衡子集（默认 4000/1000） |
| 超参 | Adam lr=1e-3、batch=64、1 epoch |
| 设备 | CPU，PyTorch 固定 `set_num_threads(1)` |
| dtype | MiniFlows float64 / PyTorch float32（各自默认，报告注明） |

## 2. 结构速览

| 符号 | 作用 |
|---|---|
| `build_miniflows_lenet` | MiniFlows 版 LeNet |
| `build_torch_lenet` | PyTorch 版 LeNet（逐层对应） |
| `_rss_mb` | 当前进程 RSS（psutil） |
| `train_miniflows` | 训练 + 指标 |
| `train_torch` | 训练 + 指标 |
| `plot_compare` | 训练时间/吞吐/延迟三子图 |
| `write_markdown` | 中文报告（含口径说明） |

## 3. 逐段讲解

### 3.1 先禁 torchvision，再导 torch

```python
os.environ.setdefault('MINIFLOWS_NO_TORCHVISION', '1')
...
X = load_mnist_arrays(...)     # 纯 numpy 路径
... import torch               # 之后才加载 torch
```

原因：torchvision 引入的 MKL 与 numpy 的 MKL 重复，在部分线程场景触发
`OMP: Error #15`。先走 numpy 数据路径可以规避。

### 3.2 同结构校验

两边参数量必须相同（44,426），这是“公平对比”的硬指标；
FLOPs 用 `profiler.count_flops` 的解析公式，PyTorch 侧直接复用同一个数
（结构相同 → FLOPs 必然相同，比较的意义是验证口径一致，而不是比谁算得少）。

### 3.3 指标

- **训练时间**：1 epoch wall time（不含数据加载）；
- **吞吐**：样本/秒 = 样本数×epochs / 时间；
- **单帧延迟**：batch=1，warmup 3 次，重复 20 次；
- **RSS 增量**：训练前后 `psutil` 读取的常驻内存差；
- **准确率**：测试集整体准确率。

### 3.4 实测（3000 样本，1 轮）

| 指标 | MiniFlows | PyTorch |
|---|---|---|
| 参数量 | 44,426 | 44,426 |
| 训练时间 | 2.27 s | 0.87 s（2.6×） |
| 吞吐 | 1321/s | 3456/s |
| 单帧延迟 | 0.96 ms | 0.46 ms（2.1×） |
| 准确率 | 0.718 | 0.725 |

比之前 `train_lenet.py` 的 8× 差距小得多：因为这里 PyTorch 固定单线程、
两边数据/更新逻辑一致。**结论随口径变化**，所以报告必须写清条件。

## 4. 产物

- `outputs/pytorch_benchmark/资源对比.json`；
- `资源对比.md`（中文表格 + 结论 + 口径说明）；
- `资源对比.png`（训练时间/吞吐/延迟三子图）。

## 5. 易错点

1. 不固定 PyTorch 线程数会得到偏小的耗时（多线程），对比就不公平；
2. dtype 不同（float64 vs float32）会让内存/时间都偏大；
3. RSS 是粗略指标，PyTorch allocator 会缓存内存；
4. 短训练（1 轮）准确率低是正常的，别写成“模型不行”；
5. 不要把“FLOPs 相同”写成“实现同样高效”——FLOPs 是结构属性。

## 6. 自检问题

1. 为什么这个脚本比 `train_lenet.py` 的差距小？
2. RSS 增量能说明什么、不能说明什么？
3. 如何让对比更公平（dtype/线程/数据）？
4. 为什么 FLOPs 要用同一公式重算？
