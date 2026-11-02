# -*- coding: utf-8 -*-
"""
miniflows/layers.py —— MiniFlows 层与容器（仿 Keras 风格）
==========================================================

【模块定位】
    本模块把 ops 中的函数式算子封装成"有状态的网络层"：每个层持有自己的
    可训练参数（param 节点），并实现 forward/__call__/params/named_params/
    train_mode/zero_grad 等统一接口；Sequential 容器把多层串成前馈网络。

【层清单】
    Linear     全连接层：y = x @ W + b（He/Xavier/普通初始化可选）
    Conv2D     二维卷积层（调用 ops.conv2d，im2col 加速）
    MaxPool2D / AvgPool2D    池化层
    Flatten    摊平层 (N,C,H,W)->(N,C*H*W)
    ReLU / Sigmoid / Tanh / Softmax    激活层
    Dropout    随机失活层（inverted dropout：训练时缩放，评估时恒等）
    Sequential 容器：逐层传递 + 参数管理 + summary 中文表格

【初始化约定（契约）】
    Linear:  He     std = sqrt(2/in)          （默认，配 ReLU）
             Xavier std = sqrt(2/(in+out))    （配 tanh/sigmoid）
             normal std = init_std
    Conv2D:  He     std = sqrt(2/(C*KH*KW))   （fan_in = C*KH*KW）

【依赖】miniflows.graph / miniflows.ops。
"""

import numpy as np

from miniflows.graph import Node, param, constant
from miniflows import ops


# ============================================================================
#                                 层基类
# ============================================================================
class Layer:
    """所有网络层的基类：规定统一接口（组合模式）。

    子类需要：
        实现 forward(x)；有参数的层重写 params()/named_params()；
        有训练/评估差异的层（如 Dropout）重写 train_mode()。
    """

    def __init__(self, name=None):
        """参数：
            name (str|None): 层名；Sequential.named_params 与 summary 用它
            作为参数名前缀，None 时用 'layer{i}'。
        """
        self.name = name
        self.training = True        # 默认训练模式（Dropout 等会用到）

    def forward(self, x):
        """前向传播：输入 Node，输出 Node。子类必须实现。"""
        raise NotImplementedError('Layer 子类必须实现 forward()')

    def __call__(self, x):
        """让层可以像函数一样调用：layer(x) 等价 layer.forward(x)。"""
        return self.forward(x)

    def params(self):
        """返回本层可训练参数列表（默认无参数层返回 []）。"""
        return []

    def named_params(self):
        """返回 [(层内参数名如 'W', Node), ...]（默认空）。"""
        return []

    def train_mode(self, flag):
        """切换训练/评估模式（默认无状态层什么都不做）。"""
        pass

    def zero_grad(self):
        """清空本层全部参数的梯度。"""
        for p in self.params():
            p.zero_grad()


# ============================================================================
#                              全连接层
# ============================================================================
class Linear(Layer):
    """全连接层 y = x @ W + b。

    参数形状：W:(in_features, out_features)，b:(out_features,)（零初始化）。
    初始化（契约）：'he'（默认）std=sqrt(2/in)；'xavier' std=sqrt(2/(in+out))；
    'normal' std=init_std。
    """

    def __init__(self, in_features, out_features, init='he',
                 init_std=0.05, name=None):
        """参数：
            in_features (int): 输入特征维；
            out_features (int): 输出特征维；
            init (str): 'he' | 'xavier' | 'normal'；
            init_std (float): init='normal' 时的标准差；
            name (str|None): 层名。
        """
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        # ---- 按契约选择初始化标准差 ----
        if init == 'he':
            # He 初始化（Kaiming）：方差 2/fan_in，配合 ReLU 保持激活方差稳定
            std = np.sqrt(2.0 / in_features)
        elif init == 'xavier':
            # Xavier/Glorot：方差 2/(fan_in+fan_out)，配合 tanh/sigmoid
            std = np.sqrt(2.0 / (in_features + out_features))
        elif init == 'normal':
            std = init_std           # 普通固定标准差
        else:
            raise ValueError(f"Linear.init 只支持 'he'|'xavier'|'normal'，收到 {init}")
        # W:(in,out) 行主序，前向直接 x @ W；b 零初始化
        self.W = param(np.random.randn(in_features, out_features) * std,
                       name='W' if name is None else f'{name}.W')
        self.b = param(np.zeros(out_features),
                       name='b' if name is None else f'{name}.b')

    def forward(self, x):
        """前向：y = x @ W + b（偏置用 broadcast_add 广播）。"""
        return ops.broadcast_add(ops.matmul(x, self.W), self.b)

    def params(self):
        """可训练参数 [W, b]。"""
        return [self.W, self.b]

    def named_params(self):
        """返回 [('W', W), ('b', b)]。"""
        return [('W', self.W), ('b', self.b)]


# ============================================================================
#                              二维卷积层
# ============================================================================
class Conv2D(Layer):
    """二维卷积层（调用 ops.conv2d，im2col 加速）。

    参数形状：W:(F, C, KH, KW)，b:(F,)（零初始化）。
    初始化：'he'（默认）std=sqrt(2/(C*KH*KW))；'xavier' std=sqrt(2/((C+F)*KH*KW))；
    'normal' std=init_std。
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, init='he', name=None):
        """参数：
            in_channels (int): 输入通道数 C；
            out_channels (int): 输出通道数（卷积核个数）F；
            kernel_size (int|tuple): 核大小；
            stride/padding: 步幅 / 零填充；
            init (str): 初始化方式；
            name (str|None): 层名。
        """
        super().__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        # 核大小支持 int 或 (kh, kw)
        if isinstance(kernel_size, (tuple, list)):
            KH, KW = int(kernel_size[0]), int(kernel_size[1])
        else:
            KH = KW = int(kernel_size)
        self.kernel_size = (KH, KW)
        # fan_in = C*KH*KW：单个输出像素看到的输入连接数
        fan_in = in_channels * KH * KW
        fan_out = out_channels * KH * KW      # fan_out 供 xavier 用
        if init == 'he':
            std = np.sqrt(2.0 / fan_in)       # He：方差 2/fan_in
        elif init == 'xavier':
            std = np.sqrt(2.0 / (fan_in + fan_out))   # Xavier
        elif init == 'normal':
            std = 0.05                        # 与 Linear 保持一致也可用 init_std
        else:
            raise ValueError(f"Conv2D.init 只支持 'he'|'xavier'|'normal'，收到 {init}")
        self.W = param(np.random.randn(out_channels, in_channels, KH, KW) * std,
                       name='W' if name is None else f'{name}.W')
        self.b = param(np.zeros(out_channels),
                       name='b' if name is None else f'{name}.b')

    def forward(self, x):
        """前向：直接委托 ops.conv2d（im2col + 矩阵乘）。"""
        return ops.conv2d(x, self.W, self.b,
                          stride=self.stride, padding=self.padding)

    def params(self):
        """可训练参数 [W, b]。"""
        return [self.W, self.b]

    def named_params(self):
        """返回 [('W', W), ('b', b)]。"""
        return [('W', self.W), ('b', self.b)]


# ============================================================================
#                                池化层
# ============================================================================
class MaxPool2D(Layer):
    """二维最大池化层（委托 ops.maxpool2d；反向只放梯度到最大值位置）。"""

    def __init__(self, kernel_size=2, stride=2):
        """参数：kernel_size 窗口大小；stride 步幅（无 padding）。"""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """前向：委托 ops.maxpool2d。"""
        return ops.maxpool2d(x, kernel_size=self.kernel_size, stride=self.stride)


class AvgPool2D(Layer):
    """二维平均池化层（委托 ops.avgpool2d）。"""

    def __init__(self, kernel_size=2, stride=2):
        """参数：kernel_size 窗口大小；stride 步幅。"""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """前向：委托 ops.avgpool2d。"""
        return ops.avgpool2d(x, kernel_size=self.kernel_size, stride=self.stride)


# ============================================================================
#                          摊平层与激活层（无参数）
# ============================================================================
class Flatten(Layer):
    """摊平层：(N,C,H,W) -> (N, C*H*W)（卷积特征图接全连接层的桥梁）。"""

    def forward(self, x):
        """前向：从 axis=1 起摊平（保留 batch 维）。"""
        return ops.flatten(x, start_axis=1)


class ReLU(Layer):
    """ReLU 激活层。"""

    def forward(self, x):
        """前向：max(0, x)。"""
        return ops.relu(x)


class Sigmoid(Layer):
    """Sigmoid 激活层。"""

    def forward(self, x):
        """前向：1/(1+e^{-x})（数值稳定实现）。"""
        return ops.sigmoid(x)


class Tanh(Layer):
    """Tanh 激活层。"""

    def forward(self, x):
        """前向：tanh(x)。"""
        return ops.tanh(x)


class Softmax(Layer):
    """Softmax 激活层（默认沿最后一维）。"""

    def __init__(self, axis=-1):
        """参数：axis: 沿哪个轴做归一化。"""
        super().__init__()
        self.axis = axis

    def forward(self, x):
        """前向：委托 ops.softmax。"""
        return ops.softmax(x, axis=self.axis)


# ============================================================================
#                                Dropout 层
# ============================================================================
class Dropout(Layer):
    """随机失活层（inverted dropout 实现）。

    训练模式：每个元素以概率 p 置零，其余除以 (1-p) 放大 ——
        mask = (rand > p) / (1-p)，y = x * mask
        这样训练/评估的期望输出一致（无需评估时再缩放）。
    评估模式：恒等映射（直接返回输入）。
    """

    def __init__(self, p=0.5):
        """参数：p (float): 丢弃概率，0<=p<1。"""
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f'Dropout.p 必须在 [0,1) 内，收到 {p}')
        self.p = p

    def train_mode(self, flag):
        """重写基类：记录当前模式供 forward 判断。"""
        self.training = bool(flag)

    def forward(self, x):
        """前向：训练时 y = x * mask（mask=(rand>p)/(1-p)）；评估时恒等。

        注意 mask 是常量 Node：它只是随机系数，不需要梯度。
        """
        if self.training and self.p > 0.0:
            # 伯努利采样：rand>p 的位置保留(1)，否则置 0；再整体除 (1-p) 补偿期望
            mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float64)
            mask /= (1.0 - self.p)            # inverted dropout 的期望补偿
            # mask 是随机系数（常量节点，不参与求导），前向只需逐元素乘
            return ops.mul(x, constant(mask, name='dropout_mask'))
        return x    # 评估模式：恒等映射


# ============================================================================
#                               Sequential 容器
# ============================================================================
class Sequential(Layer):
    """顺序容器：把多个层串成前馈网络（仿 Keras Sequential）。

    用法：model = Sequential(Linear(2,8), Tanh(), Linear(8,1))
    """

    def __init__(self, *layers):
        """参数：*layers: 任意多个 Layer 实例（可嵌套 Sequential）。"""
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        """前向：逐层传递 x。"""
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self):
        """递归收集全部子层的可训练参数。"""
        ps = []
        for layer in self.layers:
            ps.extend(layer.params())
        return ps

    def named_params(self):
        """递归收集带层级前缀的参数名。

        命名规则（契约）：前缀 = 子层 name 属性（有则用）否则 'layer{i}'，
        全名形如 'layer0.W'、'conv1.b'（嵌套时前缀逐级拼接）。
        """
        out = []
        for i, layer in enumerate(self.layers):
            prefix = layer.name if getattr(layer, 'name', None) else f'layer{i}'
            for pname, p in layer.named_params():
                out.append((f'{prefix}.{pname}', p))
        return out

    def train_mode(self, flag):
        """递归设置全部子层的训练/评估模式。"""
        for layer in self.layers:
            layer.train_mode(flag)

    def summary(self, input_shape):
        """打印模型结构摘要（中文表格）：每层名/输出形状/参数量/总参数量。

        参数：
            input_shape (tuple): 不含 batch 维的输入形状，如 (1, 28, 28)。
        返回：
            int: 总参数量。

        实现思路：
            用 batch=1 的全零输入逐层真实前向一遍（评估模型各层输出形状），
            既能覆盖卷积/池化的尺寸计算，也能适配任意自定义层。
        """
        shape = tuple(int(d) for d in input_shape)
        x_np = np.zeros((1,) + shape)          # batch=1 的哑输入
        rows = []
        total = 0
        for i, layer in enumerate(self.layers):
            out = layer(Node(x_np))            # 真实前向拿输出形状
            out_shape = out.data.shape
            x_np = out.data                    # 作为下一层的输入
            n_param = sum(p.data.size for p in layer.params())
            total += n_param
            lname = getattr(layer, 'name', None) or type(layer).__name__
            rows.append((i, lname, out_shape, n_param))
        # ---------- 打印中文表格 ----------
        line = '=' * 62
        print(line)
        print(' Sequential 模型结构摘要')
        print(line)
        print(f"{'序号':<6}{'层名':<16}{'输出形状':<22}{'参数量':>10}")
        print('-' * 62)
        for i, lname, out_shape, n_param in rows:
            print(f'{i:<6}{lname:<16}{str(out_shape):<22}{n_param:>10}')
        print('-' * 62)
        print(f'总参数量: {total:,}')
        print(line)
        return total
