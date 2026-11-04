# `experiments/exp2_grad_check.py` 阅读指南

> 前置：`autodiff.py`、各算子 ｜ 难度：★★ ｜ 建议 40 分钟 ｜ 扩展②

## 1. 文件定位

对 16 个算子 + 1 个 MLP 做中心差分数值梯度检验，输出中文 Markdown 报告。
这是“框架实现正确”的最硬证据。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `make_away_from_zero` (67) | 构造远离不可导点的输入 |
| 2 | `make_pool_input` (84) | 构造 maxpool 无平局输入 |
| 3 | `build_checks` (113) | 16 个算子的 checks 列表 |
| 4 | `build_mlp_check` (267) | MLP+CE 全参数检查 |
| 5 | `main` (290) | 逐项运行 + 打印表格 |
| 6 | `write_report` (343) | 写 `outputs/exp2/梯度检验报告.md` |

## 3. 逐段讲解

### 3.1 输入构造

- ReLU 输入避开 0 附近；
- maxpool 输入用随机不重复值，避免并列最大值；
- log/sqrt 输入 >0；
- matmul/conv 用小型随机张量。

### 3.2 checks 列表

每项是 `dict(name, desc, arrays, f)`：

```python
f=lambda ns: ops.sum(ops.mul(ops.conv2d(ns[0], ns[1], ns[2]), W))
```

- 用“输出 × 随机权重再求和”标量化，保证每个输入元素都有非平凡梯度；
- `gradient_check` 内部做数值 vs 解析。

### 3.3 17 项与最大误差

| # | 算子 | max_rel |
|---|---|---|
| 1 | add | 3.4e-11 |
| 2 | matmul | 1.1e-10 |
| 3 | broadcast_add | 2.8e-10 |
| 4 | mul | 9.4e-11 |
| 5 | relu | 4.4e-11 |
| 6 | sigmoid | 3.5e-10 |
| 7 | tanh | **1.361e-7** |
| 8 | softmax | 1.6e-10 |
| 9 | MSE | 1.5e-10 |
| 10 | BCE | 2.0e-10 |
| 11 | CE | 1.2e-9 |
| 12 | conv2d | 1.9e-9 |
| 13 | maxpool | 5.5e-10 |
| 14 | avgpool | 2.5e-11 |
| 15 | transpose | 8.0e-9 |
| 16 | sum | 3.6e-11 |
| 17 | MLP+CE | 1.9e-9 |

全部 <1e-5。注意：**exp2 的最大值是 1.361e-7（tanh）**；
2.978e-9 是 `test_core.py` 中 34 项的最大值，两个口径不要混写。

### 3.4 报告

`梯度检验报告.md` 包含：生成时间、eps/阈值、逐项表格、结论
（中心差分原理、误差来源、为什么 float64）。

## 4. 调用关系

`autodiff.gradient_check → ops/模型 → 表格 → write_report`；
`tests/test_core.py` 复用同一套工具，但检查项更多（34 项）。

## 5. 易错点

1. 不可导点导致误报（ReLU 0、maxpool 平局）；
2. eps 太大/太小都会增大误差；
3. 输入必须是 float64；
4. 报告数字与终端输出要一致；
5. 不要把 exp2 的 17 项与 test_core 的 34 项混为一谈。

## 6. 自检问题

1. 为什么用“输出×随机权重求和”标量化？
2. 如何构造一个“已知错误反向”验证工具能报错？
3. tanh 的误差为什么比其他算子大？
