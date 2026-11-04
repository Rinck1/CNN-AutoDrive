# `miniflows/layers.py` 阅读指南

> 前置：`graph.py`、`ops.py` ｜ 难度：★★ ｜ 建议 60 分钟 ｜ 核心

## 1. 文件定位

把算子封装成“层”：参数管理、初始化、前向组合、训练/评估模式。
`Sequential` 是最常用的模型容器，应用脚本里的 `build_model()` 大多是它。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `Layer` (38) | 层基类：`forward/__call__/params/named_params/train_mode/zero_grad` |
| 2 | `Linear` (83) | 全连接：`y=xW+b` |
| 3 | `Conv2D` (136) | 卷积层：包装 `ops.conv2d` |
| 4 | `MaxPool2D` (198) / `AvgPool2D` (212) | 池化 |
| 5 | `Flatten` (229) | `(N,C,H,W)→(N,C*H*W)` |
| 6 | 激活层 (237~275) | ReLU/Sigmoid/Tanh/Softmax |
| 7 | `Dropout` (277) | inverted dropout |
| 8 | `Sequential` (314) | 顺序容器 + 参数量统计 + summary |

## 3. 逐段讲解

### 3.1 Layer 基类

统一接口：

```python
class Layer:
    def forward(self, x): ...
    def __call__(self, x): return self.forward(x)
    def params(self): return []
    def named_params(self): return []
    def train_mode(self, flag): pass
    def zero_grad(self): ...
```

组合模式：有参数的层重写 `params/named_params`，有训练态差异的层重写 `train_mode`。

### 3.2 Linear

```python
W:(in_features, out_features)   # He: std=sqrt(2/in)
b:(out_features,) = 0
y = x @ W + b
```

- W 形状是 `(in,out)`，不是 `(out,in)`；
- 支持 2D `(N,in)`、3D `(N,L,in)`；3D 偏置对齐靠 `broadcast_add` 的最后一维规则；
- `named_params()` 返回 `[('W', W), ('b', b)]`。

### 3.3 Conv2D

包装 `ops.conv2d`；参数形状：

```python
W:(out_channels, in_channels, KH, KW)
b:(out_channels,)
```

He 初始化 `std=sqrt(2/(C*KH*KW))`；`stride/padding` 原样传给算子。

### 3.4 池化 / Flatten / 激活

- MaxPool2D / AvgPool2D 只是转发 ops；
- Flatten 把第 1 维之后全部展平（默认 start_axis=1）；
- 激活层无参数，直接调用 ops。

### 3.5 Dropout

训练态：

```python
mask = (rand > p) / (1-p)
y = x * mask
```

评估态：恒等。`inverted dropout` 保证训练/推理期望一致，推理时不用再缩放。

### 3.6 Sequential

- `forward`：逐层传递；
- `params`：递归拼接；
- `named_params`：前缀 = 层 `name` 或 `layer{i}`，嵌套时逐级拼接；
- `train_mode`：递归；
- `summary(input_shape)`：用 batch=1 的零输入真实前向一遍，打印每层输出形状/参数量。

命名示例：

```python
Sequential(Conv2D(1,4,3,name='conv1'), ReLU(), Linear(16*8*8,10))
# named_params -> conv1.W, conv1.b, layer2.W, layer2.b
```

注意 Linear 在容器里是第 2 层，所以是 `layer2`，不是 `layer1`。

## 4. 调用关系

所有模型（CNN/ViT 的 block 除外）都由层组合；`Trainer` 调 `model(x)`、
`model.params()`、`model.train_mode()`；`utils.save_model` 用 `named_params()`。

## 5. 易错点

1. **W 形状**：`(in,out)`，`get_w_b()` 读取时要按列向量处理；
2. **Linear 的 3D 输入**：偏置必须对齐最后一维，4D 卷积偏置才按通道；
3. **命名**：`layer{i}` 用的是容器索引，不是“第几个带参层”；
4. **summary 的输入形状**：不含 batch 维，例如 `(1,28,28)`；
5. **Dropout 评估态**：`Trainer.evaluate` 会自动 `train_mode(False)`，手动推理要自己关。

## 6. 自检问题

1. `Linear(4,3)` 的 W、b 形状与参数量？
2. He 初始化的 std 公式？为什么除以 fan_in？
3. `Sequential` 里 Dropout 如何知道当前是训练还是评估？
4. `summary` 为什么用真实前向而不是手工推导形状？
5. 为什么 `named_params` 要保证名字唯一？
