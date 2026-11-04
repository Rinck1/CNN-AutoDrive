# `miniflows/resnet.py` 阅读指南

> 前置：`ops.py`（卷积/池化/归约）、`layers.py` ｜ 难度：★★★ ｜ 建议 60 分钟

## 1. 文件定位

在 MiniFlows 上实现 ResNet（BasicBlock + 残差连接），用于 DonkeyCar 转向回归：

```
Stem: Conv(1→8,k5,s2,p2) → GroupNorm → ReLU       64→32
Stage1: 2×BasicBlock(8→8, s1)                      32×32
Stage2: 2×BasicBlock(8→16, s2)                     16×16
Stage3: 2×BasicBlock(16→32, s2)                    8×8
GlobalAvgPool → Linear(32→1) → Tanh
```

参数量 44,337，与 CNN 44,426 / ViT 44,353 基本对齐。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `GroupNorm2D` (48) | 分组归一化（替代 BatchNorm） |
| 2 | `_make_gn` (约 96) | 自动选能整除通道数的组数 |
| 3 | `BasicBlock` (约 104) | 两个 3×3 conv + 残差 |
| 4 | `_prefixed` | 参数名前缀 |
| 5 | `ResNetRegressor` (约 170) | Stem + 3 stage + 全局池化 + 回归头 |
| 6 | `build_resnet` (约 250) | 默认配置工厂 |

## 3. 逐段讲解

### 3.1 为什么用 GroupNorm 而不是 BatchNorm

- MiniFlows 的 `save_model` 只保存 `named_params()`，没有 BN 的 running_mean/var；
- GroupNorm 不依赖 batch 统计，训练/评估行为一致，保存/加载只存参数即可逐位复现；
- 小 batch（32）下 GroupNorm 通常比 BN 稳定。
- 代价：推理时每组都要算均值和方差，略慢。

**结论**：这是“为了当前框架的持久化语义”做的工程取舍，不是标准 ResNet-18 的完整复刻。

### 3.2 GroupNorm2D

```python
x:(N,C,H,W) → reshape (N,G,C/G,H,W)
μ,σ² = mean/var over (C/G,H,W)
y = γ·(x-μ)/√(σ²+ε)+β
reshape 回 (N,C,H,W)
```

- `γ/β` 形状 `(C,)`，用 `ops.reshape` 补成 `(1,C,1,1)` 再广播；
- 反向完全由 `mean/var/sqrt/div` 自动求导；
- `train_mode` 无状态（与 Dropout 不同）。

### 3.3 BasicBlock

```python
identity = x if 形状不变 else 1×1 Conv + GroupNorm
y = ReLU(GN(Conv3x3(ReLU(GN(Conv3x3(x))))))
return ReLU(y + identity)
```

- 当 `stride != 1` 或通道数变化时，shortcut 必须做 1×1 投影，否则形状无法相加；
- 第二个 GN 后**先加残差再 ReLU**，这是 ResNet v1 的 post-activation 形式；
- `named_params` 前缀 `conv1/gn1/conv2/gn2/shortcut`。

### 3.4 ResNetRegressor

- `stem` 用 5×5 stride 2，把 64×64 降到 32×32，减少后续计算量；
- 三个 stage 分别下采样到 32/16/8；
- `ops.mean(x, axis=(2,3))` 做全局平均池化，得到 `(N,C)`；
- 回归头 `Linear(C→1)+Tanh`，输出转向角 ∈[-1,1]。

### 3.5 参数量

默认 C0=8、widths=(8,16,32)、blocks=(2,2,2)：

- stem 约 200；
- stage1 约 2.3k；
- stage2 约 8.1k；
- stage3 约 32.3k；
- 加上 GN/head ≈ **44,337**。

## 4. 调用关系

`applications/donkeycar_resnet.py` 调 `build_resnet`；
`tests/test_resnet.py` 验证 GroupNorm 统计/梯度、极小 ResNet 全参数梯度、保存加载。

## 5. 易错点

1. **GroupNorm 组数必须整除通道数**：`_make_gn` 会自动降组数；
2. **shortcut 形状**：stride/通道变化必须投影；
3. **relu 顺序**：第二个 GN 之后先加残差再 ReLU；
4. **全局池化轴**：`axis=(2,3)`，不是 `(0,2,3)`；
5. **不要假设有 BN running stats**：GroupNorm 没有，保存/加载只存参数。

## 6. 自检问题

1. 画出 BasicBlock 的数据流与形状；
2. 为什么 shortcut 在通道变化时需要 1×1 卷积？
3. GroupNorm 与 BatchNorm 的训练/评估差异？
4. 默认 ResNet 的 44,337 参数如何拆解？
