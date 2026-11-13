# `miniflows/classic_cnn.py` 阅读指南

> 前置：`ops.py`（conv2d/分组卷积）、`resnet.py`（GroupNorm） ｜ 难度：★★★ ｜ 建议 60 分钟

## 1. 文件定位

补齐课程要求的经典 CNN：**VGG11 风格**与 **MobileNetV1**（深度可分离卷积），
与 `resnet.py` 一起覆盖 “VGG / ResNet / MobileNet” 三个模型族。

## 2. 结构速览

| 顺序 | 符号 | 作用 |
|---|---|---|
| 1 | `DepthwiseConv2D` | 逐通道卷积，W:(C,1,KH,KW) |
| 2 | `SeparableConv2D` | dw → GN → ReLU → 1×1 pw → GN → ReLU |
| 3 | `VGG11` | 5 个卷积块（16/32/64/64/64×width）+ 全局平均池化 |
| 4 | `_SequentialNamed` | 内部带名字的顺序容器 |
| 5 | `MobileNetV1` | stem + 13 个深度可分离块 + 全局平均池化 |
| 6 | `build_vgg11/build_mobilenet_v1/build_classic_model` | 工厂 |

## 3. 逐段讲解

### 3.1 深度可分离卷积为什么省

标准 3×3 卷积乘加量：`OH·OW·F·C·9`；
深度可分离拆成：
```
逐通道卷积：OH·OW·C·9
1×1 逐点卷积：OH·OW·F·C
```
比值约 `1/F + 1/9`。实测 32→64 通道时，可分离 2,432 参数 vs 标准 18,496（0.13×）。

### 3.2 DepthwiseConv2D 的实现选择

内部调用 `ops.depthwise_conv2d`，而不是 `ops.conv2d_groups(groups=C)`：
后者会退化成 C 次 `im2col`，在 MobileNet（C 最大 1024）上非常慢。
向量化实现用 `KH×KW` 次 strided-slice 乘加，前向从 53ms 降到 22ms。

### 3.3 VGG11 的改造

标准 VGG16 的 3 层 4096 全连接有上亿参数，纯 numpy 不可训练。这里：

- 通道按 `width` 缩放（默认 16/32/64/64/64）；
- 5 个卷积块 + 全局平均池化 + 单层 Linear；
- GroupNorm 替代 BatchNorm（无 running stats，保存/加载一致）。

保留的是 VGG 的核心特征：**连续 3×3 小卷积堆叠 + 逐块池化**。

### 3.4 MobileNetV1 的结构

```
stem: Conv(1→32w, 3, s2) + GN + ReLU
13 × SeparableConv（通道 64/128/128/256/…/1024 × width，s2 共 5 次）
GlobalAvgPool → Linear
```

64×64 输入经 5 次 stride-2 后为 2×2。

### 3.5 参数量参考（64×64）

| 模型 | 参数 | FLOPs |
|---|---:|---:|
| VGG11（w=1.0） | 208,865 | 51.7 M |
| MobileNetV1（w=0.25） | 215,921 | 23.4 M |
| MobileNetV1（w=1.0） | 3.22 M | — |

注意：MobileNet 参数量更小，但纯 numpy 下延迟不一定更低——
它的层数多、单层张量小，Python 调度开销占比高。这是“FLOPs ≠ 实际延迟”的典型例子。

## 4. 调用关系

`applications/mnist_web.py --model vgg11/mobilenet` 调用；
`tests/test_models.py` 校验前向形状、参数效率与深度卷积梯度；
`profiler.profile_model` 读取参数量/FLOPs。

## 5. 易错点

1. 深度卷积要求 `img_size` 能被 2^5 整除（5 次下采样）；
2. GroupNorm 组数必须整除通道数（`_make_gn` 自动降组）；
3. VGG 的 `width` 太小（<0.25）会让通道数过少、精度明显下降；
4. MobileNet 的 `width=1.0` 有 320 万参数，CPU 训练很慢，演示建议 0.25；
5. `build_classic_model('resnet')` 只转发 resnet.py，不接受 `num_classes`。

## 6. 自检问题

1. 推导深度可分离卷积的乘加量比值；
2. 为什么不用 `conv2d_groups(groups=C)` 实现深度卷积？
3. VGG 改造丢掉了什么、保留了什么？
4. 为什么 MobileNet 的 FLOPs 低但 numpy 延迟不低？
