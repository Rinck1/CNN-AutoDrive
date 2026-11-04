# `pytorch_benchmark/lenet5.py` 阅读指南

> 前置：PyTorch 基础 ｜ 难度：★ ｜ 建议 20 分钟 ｜ 对照

## 1. 文件定位

用 PyTorch 实现与自研 MNIST 版**完全同结构**的 LeNet-5，作为公平对照。
参数量必须与 MiniFlows 版一致：**44,426**。

## 2. 结构速览

| 符号/行号 | 作用 |
|---|---|
| `LeNet5` (57) | `nn.Module`，逐层注释尺寸 |
| `count_parameters` (124) | 统计可训练参数 |

## 3. 逐段讲解

### 3.1 结构

```
Conv2d(1,6,5) → ReLU → MaxPool2d(2)
Conv2d(6,16,5) → ReLU → MaxPool2d(2)
Flatten → Linear(256,120) → ReLU
Linear(120,84) → ReLU
Linear(84,10)
```

与 `applications/mnist.py:build_lenet` 一一对应；尺寸推导：

```
28→24→12→8→4；16*4*4=256
```

### 3.2 参数量

```
Conv1: 6*1*5*5+6=156
Conv2: 16*6*5*5+16=2416
FC1: 256*120+120=30840
FC2: 120*84+84=10164
FC3: 84*10+10=850
总: 44426
```

### 3.3 为什么单独一个文件

- 训练脚本 `train_lenet.py` 负责数据/训练/对比；
- 模型定义独立，便于核对结构与参数量；
- 不引入 MiniFlows 依赖。

## 4. 调用关系

`train_lenet.py` import `LeNet5`，构建模型、训练、评估。

## 5. 易错点

1. 层序/通道数与自研版必须完全一致；
2. Flatten 后是 256，不是 16*6*6；
3. 参数量 44,426 是对比公平性的硬指标；
4. 不要在模型里加 BatchNorm/Dropout，否则对比不公平。

## 6. 自检问题

1. 逐层手算输出尺寸与参数量；
2. 如果自研版多一个 Conv，对比会怎样？
3. 为什么对照组要固定 CPU？
