# -*- coding: utf-8 -*-
"""
miniflows/classic_cnn.py —— 经典 CNN 模型库（VGG / MobileNet）
================================================================

课程要求“框架至少能训练、评估、可视化、部署经典的 CNN 模型
（VGG / ResNet / MobileNet）”，ResNet 见 `miniflows/resnet.py`，
本文件补齐 VGG 与 MobileNet，并提供统一工厂。

    build_vgg11(...)       VGG11 风格（可缩放宽度），适合 32×32/64×64 输入
    build_mobilenet_v1(...) MobileNetV1（深度可分离卷积），参数量最小
    build_resnet(...)      见 resnet.py（这里只做转发，便于统一调用）

设计取舍
--------
1. **用 GroupNorm 代替 BatchNorm**：与 resnet.py 一致，框架暂无 BN 的
   running stats 持久化；GroupNorm 无状态，save_model/load_model 可逐位复现。
2. **VGG 用全局平均池化替代 4096 全连接**：标准 VGG16 的 FC 层有上亿参数，
   在 CPU + 纯 numpy 下不可训练；这里保留 5 个卷积块的结构，用全局平均池化
   + 1 层分类头，参数量降到 1~3 万量级，仍保留“深而窄、3×3 堆叠”的 VGG 特征。
3. **MobileNet 实现真正的深度可分离卷积**：`ops.conv2d_groups(groups=C)` 做
   逐通道卷积，再接 1×1 逐点卷积；这是它参数量/计算量小的原因。
"""

import numpy as np

from miniflows.graph import Node, param
from miniflows.layers import Layer, Linear, Conv2D, MaxPool2D, Flatten, ReLU
from miniflows.resnet import GroupNorm2D, _make_gn
from miniflows import ops


# ======================================================================
# 一、深度可分离卷积
# ======================================================================
class DepthwiseConv2D(Layer):
    """逐通道卷积（depthwise）：每个输入通道用自己的 KH×KW 核卷积，不做通道混合。

    参数形状：W:(C, 1, KH, KW)，b:(C,)。
    实现在 ops.conv2d_groups 上：groups=C，F=C。
    """

    def __init__(self, channels, kernel_size=3, stride=1, padding=1,
                 init='he', name=None):
        """参数：通道数、核大小、步幅、填充、初始化方式、层名。"""
        super().__init__(name)
        self.channels = int(channels)
        self.stride = int(stride)
        self.padding = int(padding)
        if isinstance(kernel_size, (tuple, list)):
            KH, KW = int(kernel_size[0]), int(kernel_size[1])
        else:
            KH = KW = int(kernel_size)
        self.kernel_size = (KH, KW)
        # 每个通道只看自己那一个输入通道 → fan_in = KH*KW
        std = np.sqrt(2.0 / (KH * KW)) if init == 'he' else 0.05
        self.W = param(np.random.randn(channels, 1, KH, KW) * std, name='W')
        self.b = param(np.zeros(channels), name='b')

    def forward(self, x):
        """x:(N,C,H,W) -> (N,C,OH,OW)（向量化深度卷积）。"""
        return ops.depthwise_conv2d(x, self.W, self.b, stride=self.stride,
                                    padding=self.padding)

    def named_params(self):
        """返回 [('W', W), ('b', b)]。"""
        return [('W', self.W), ('b', self.b)]

    def params(self):
        """可训练参数。"""
        return [self.W, self.b]


class SeparableConv2D(Layer):
    """深度可分离卷积 = DepthwiseConv2D + 1×1 Pointwise Conv。

    MobileNet 的基本单元；相比标准 3×3 卷积，乘加量约为 1/C + 1/9。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, num_groups=None, name=None):
        """参数：输入/输出通道、核大小、步幅、填充、GroupNorm 组数。"""
        super().__init__(name)
        self.dw = DepthwiseConv2D(in_channels, kernel_size, stride, padding,
                                  name='dw')
        self.gn1 = _make_gn(in_channels, num_groups)
        self.pw = Conv2D(in_channels, out_channels, 1, stride=1, padding=0,
                         name='pw')
        self.gn2 = _make_gn(out_channels, num_groups)

    def forward(self, x):
        """x:(N,Cin,H,W) -> (N,Cout,OH,OW)，两次 GN+ReLU。"""
        x = ops.relu(self.gn1(self.dw(x)))     # 逐通道卷积 + 归一化 + 激活
        x = ops.relu(self.gn2(self.pw(x)))     # 1×1 通道混合
        return x

    def named_params(self):
        """dw / gn1 / pw / gn2 的参数。"""
        out = []
        out.extend([(f'dw.{k}', p) for k, p in self.dw.named_params()])
        out.extend([(f'gn1.{k}', p) for k, p in self.gn1.named_params()])
        out.extend([(f'pw.{k}', p) for k, p in self.pw.named_params()])
        out.extend([(f'gn2.{k}', p) for k, p in self.gn2.named_params()])
        return out

    def params(self):
        """可训练参数。"""
        return [p for _, p in self.named_params()]

    def train_mode(self, flag):
        """递归切换模式。"""
        for layer in (self.dw, self.gn1, self.pw, self.gn2):
            layer.train_mode(flag)


# ======================================================================
# 二、VGG11 风格
# ======================================================================
class VGG11(Layer):
    """VGG11 风格网络：5 个卷积块（每块 1~2 个 3×3 卷积 + 池化），全局平均池化 + 分类头。

    与标准 VGG11 的差异（为了 CPU + numpy 可训练）：
        - 通道数按 `width` 缩放（默认 16/32/64/64/64 而非 64/128/256/512/512）；
        - 用全局平均池化替代 3 层 4096 全连接（否则参数量上亿）；
        - 用 GroupNorm 替代 BatchNorm（无 running stats）。

    输入要求：H、W 能被 2^5=32 整除（5 次池化），如 64×64 或 32×32。
    """

    def __init__(self, in_channels=1, num_classes=10, width=1.0,
                 num_groups=None, name='vgg11'):
        super().__init__(name)
        base = [int(round(c * width)) for c in (16, 32, 64, 64, 64)]
        cfg = [(base[0], 1), (base[1], 1), (base[2], 2),
               (base[3], 2), (base[4], 2)]        # (通道数, 该块卷积层数)
        self.blocks = []
        in_ch = in_channels
        for bi, (out_ch, n_conv) in enumerate(cfg):
            layers = []
            for ci in range(n_conv):
                layers.append(Conv2D(in_ch, out_ch, 3, padding=1,
                                     name=f'conv{ci}'))
                layers.append(_make_gn(out_ch, num_groups))
                layers.append(ReLU())
                in_ch = out_ch
            layers.append(MaxPool2D(2, 2))
            block = _SequentialNamed(layers, name=f'block{bi}')
            self.blocks.append(block)
        self.head = Linear(in_ch, num_classes, name='head')

    def forward(self, x):
        """x:(N,C,H,W) -> (N,num_classes) logits。"""
        for block in self.blocks:
            x = block(x)
        x = ops.mean(x, axis=(2, 3))          # 全局平均池化 -> (N,C)
        return self.head(x)

    def named_params(self):
        """扁平参数名（block{i}.conv{j}.W 等）。"""
        out = []
        for i, block in enumerate(self.blocks):
            out.extend([(f'block{i}.{k}', p) for k, p in block.named_params()])
        out.extend([(f'head.{k}', p) for k, p in self.head.named_params()])
        return out

    def params(self):
        """可训练参数。"""
        return [p for _, p in self.named_params()]

    def train_mode(self, flag):
        """递归切换模式。"""
        for block in self.blocks:
            block.train_mode(flag)
        self.head.train_mode(flag)


class _SequentialNamed(Layer):
    """内部小工具：带名字的顺序容器（避免依赖 Sequential 的索引命名）。"""

    def __init__(self, layers, name=None):
        super().__init__(name)
        self.layers = list(layers)

    def forward(self, x):
        """逐层前向。"""
        for layer in self.layers:
            x = layer(x)
        return x

    def named_params(self):
        """按层类型自动生成参数名（同名层加序号区分）。"""
        out = []
        counts = {}
        for layer in self.layers:
            cls = type(layer).__name__.lower()
            idx = counts.get(cls, 0)
            counts[cls] = idx + 1
            for k, p in layer.named_params():
                out.append((f'{cls}{idx}.{k}', p))
        return out

    def params(self):
        """可训练参数。"""
        return [p for _, p in self.named_params()]

    def train_mode(self, flag):
        """递归切换模式。"""
        for layer in self.layers:
            layer.train_mode(flag)


# ======================================================================
# 三、MobileNetV1
# ======================================================================
class MobileNetV1(Layer):
    """MobileNetV1：1 个标准卷积 + 13 个深度可分离卷积 + 全局平均池化 + 分类头。

    与论文的差异：通道数按 `width` 缩放；GroupNorm 替代 BN；输入 64×64 时
    stride-2 层共 5 次，输出 2×2 特征图。
    """

    def __init__(self, in_channels=1, num_classes=10, width=1.0,
                 num_groups=None, name='mobilenet'):
        super().__init__(name)
        def w(c):
            """按宽度系数缩放通道数（至少 8）。"""
            return max(8, int(round(c * width)))

        # stem：普通 3×3 卷积
        self.stem = _SequentialNamed([
            Conv2D(in_channels, w(32), 3, stride=2, padding=1, name='conv'),
            _make_gn(w(32), num_groups),
            ReLU(),
        ], name='stem')
        # 13 个深度可分离卷积（(输出通道, 步幅) 与论文一致）
        cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
               (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
               (1024, 2), (1024, 1)]
        self.blocks = []
        in_ch = w(32)
        for i, (out_c, s) in enumerate(cfg):
            oc = w(out_c)
            self.blocks.append(SeparableConv2D(in_ch, oc, 3, stride=s,
                                               padding=1,
                                               num_groups=num_groups,
                                               name=f'sep{i}'))
            in_ch = oc
        self.head = Linear(in_ch, num_classes, name='head')

    def forward(self, x):
        """x:(N,C,H,W) -> (N,num_classes) logits。"""
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = ops.mean(x, axis=(2, 3))
        return self.head(x)

    def named_params(self):
        """扁平参数名（stem.* / sep{i}.* / head.*）。"""
        out = [(f'stem.{k}', p) for k, p in self.stem.named_params()]
        for i, block in enumerate(self.blocks):
            out.extend([(f'sep{i}.{k}', p) for k, p in block.named_params()])
        out.extend([(f'head.{k}', p) for k, p in self.head.named_params()])
        return out

    def params(self):
        """可训练参数。"""
        return [p for _, p in self.named_params()]

    def train_mode(self, flag):
        """递归切换模式。"""
        self.stem.train_mode(flag)
        for block in self.blocks:
            block.train_mode(flag)
        self.head.train_mode(flag)


# ======================================================================
# 四、统一工厂
# ======================================================================
def build_vgg11(in_channels=1, num_classes=10, width=1.0, num_groups=None):
    """构造 VGG11 风格模型（默认 width=1.0 时约 1.9 万参数，适合 64×64 输入）。"""
    return VGG11(in_channels=in_channels, num_classes=num_classes,
                 width=width, num_groups=num_groups)


def build_mobilenet_v1(in_channels=1, num_classes=10, width=1.0,
                       num_groups=None):
    """构造 MobileNetV1（width=0.25 时约 1 万参数，CPU 可训练）。"""
    return MobileNetV1(in_channels=in_channels, num_classes=num_classes,
                       width=width, num_groups=num_groups)


def build_classic_model(name, in_channels=1, num_classes=10, width=1.0):
    """按名字构造经典模型：'lenet' | 'vgg11' | 'mobilenet' | 'resnet'。

    'lenet' 返回 None（由调用方自建，因为结构依赖输入尺寸），
    其余返回 Layer 对象。
    """
    key = name.lower()
    if key in ('vgg', 'vgg11'):
        return build_vgg11(in_channels, num_classes, width)
    if key in ('mobilenet', 'mobilenetv1', 'mobilenet_v1'):
        return build_mobilenet_v1(in_channels, num_classes, width)
    if key in ('resnet', 'resnet18'):
        from miniflows.resnet import build_resnet
        return build_resnet(in_channels=in_channels)
    raise ValueError(f'未知模型名: {name}（支持 vgg11 / mobilenet / resnet）')
