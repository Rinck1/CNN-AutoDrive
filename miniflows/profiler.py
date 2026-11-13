# -*- coding: utf-8 -*-
"""
miniflows/profiler.py —— 模型规模、FLOPs、推理延迟与内存分析
==============================================================

课程要求“与其他框架（PyTorch/TensorFlow/Paddle）比较训练时间、资源占用、
FLOPs 等指标”。本模块提供统一口径的度量工具：

    count_params(model)                         参数量（与 utils 一致）
    count_flops(model, input_shape)             乘加次数（FLOPs，2×MACs）
    measure_latency(model, input_shape, ...)    单帧延迟 / 吞吐
    measure_activation_bytes(model, input_shape) 前向峰值激活内存（估算）
    profile_model(model, input_shape)           一次性返回上述四项

FLOPs 计算口径
--------------
    Conv2D / 分组卷积：2 · N · OH · OW · F · C_g · KH · KW   （乘 + 加各算 1）
    Linear：           2 · N · in · out
    GroupNorm：        ≈ 5 · N · C · H · W（均值/方差/归一化/仿射）
    Pool / 激活：      ≈ 1 · N · C · OH · OW（逐元素）
    注意：不同框架/论文对 FLOPs 的定义（是否计 bias、是否把乘加算 2）不完全一致，
    所以**跨框架比较必须在同一公式下重算**，不能直接抄别人的数字。

激活内存估算
------------
    只累加**单次前向**各层输出张量大小（float64 = 8 字节），不含梯度。
    训练内存还要加上反向缓存（约为前向的 2~3 倍），本工具给出下界。
"""

import time

import numpy as np

from miniflows.graph import Node
from miniflows.layers import (Conv2D, Linear, MaxPool2D, AvgPool2D, Flatten,
                              ReLU, Sigmoid, Tanh, Softmax, Dropout)
from miniflows.resnet import GroupNorm2D, BasicBlock, _make_gn
from miniflows.transformer import (LayerNorm, GELU, MultiHeadSelfAttention,
                                   TransformerBlock, PatchEmbedding)


def _out_hw(h, w, kh, kw, stride, padding):
    """卷积/池化输出尺寸。"""
    return ((h + 2 * padding - kh) // stride + 1,
            (w + 2 * padding - kw) // stride + 1)


def _flops_of(layer, shape, batch):
    """计算单层的 FLOPs 与输出形状。

    参数：
        layer (Layer): 层对象；
        shape (tuple): 输入形状（不含 batch）；
        batch (int): batch 大小。

    返回：
        (flops, out_shape)；无法识别的层返回 (0, 同形状)。
    """
    N = batch
    # ---- 卷积 ----
    if isinstance(layer, Conv2D):
        C, H, W = shape
        KH, KW = layer.kernel_size
        OH, OW = _out_hw(H, W, KH, KW, layer.stride, layer.padding)
        flops = 2 * N * OH * OW * layer.out_channels * C * KH * KW
        return flops, (layer.out_channels, OH, OW)
    # ---- 深度可分离卷积（在 classic_cnn 里定义，按属性识别，避免循环导入）----
    cls_name = type(layer).__name__
    if cls_name == 'DepthwiseConv2D':
        C, H, W = shape
        KH, KW = layer.kernel_size
        OH, OW = _out_hw(H, W, KH, KW, layer.stride, layer.padding)
        return 2 * N * OH * OW * C * KH * KW, (C, OH, OW)
    if cls_name == 'SeparableConv2D':
        # dw + pw 两部分
        f1, s1 = _flops_of(layer.dw, shape, N)
        f2, s2 = _flops_of(layer.pw, s1, N)
        return f1 + f2, s2
    # ---- 全连接 ----
    if isinstance(layer, Linear):
        flat = int(np.prod(shape))
        out_shape = (layer.named_params()[0][1].data.shape[-1],)
        return 2 * N * flat * out_shape[0], out_shape
    # ---- 池化 ----
    if isinstance(layer, (MaxPool2D, AvgPool2D)):
        C, H, W = shape
        k = layer.kernel_size
        kh, kw = (k, k) if isinstance(k, int) else (int(k[0]), int(k[1]))
        OH, OW = _out_hw(H, W, kh, kw, layer.stride, 0)
        return N * C * kh * kw * OH * OW, (C, OH, OW)
    # ---- 归一化 ----
    if isinstance(layer, GroupNorm2D):
        C, H, W = shape
        return 5 * N * C * H * W, shape
    if isinstance(layer, LayerNorm):
        D = shape[-1]
        return 5 * N * int(np.prod(shape[:-1])) * D, shape
    # ---- Flatten / 激活 / Dropout ----
    if isinstance(layer, Flatten):
        return 0, (int(np.prod(shape)),)
    if isinstance(layer, (ReLU, Sigmoid, Tanh, GELU, Softmax, Dropout)):
        return N * int(np.prod(shape)), shape
    # ---- 复合模块：递归 ----
    if isinstance(layer, BasicBlock):
        f1, s1 = _flops_of(layer.conv1, shape, N)
        f2, s2 = _flops_of(layer.gn1, s1, N)
        f3, s3 = _flops_of(layer.conv2, s2, N)
        f4, s4 = _flops_of(layer.gn2, s3, N)
        total = f1 + f2 + f3 + f4
        if layer.shortcut is not None:
            f5, s5 = _flops_of(layer.shortcut.layers[0], shape, N)
            f6, _ = _flops_of(layer.shortcut.layers[1], s5, N)
            total += f5 + f6
        return total, s4
    if isinstance(layer, MultiHeadSelfAttention):
        D = layer.dim
        L = shape[-2] if len(shape) >= 2 else 1
        # Q/K/V/O 四个投影 + 注意力矩阵乘 + 加权求和
        proj = 4 * 2 * N * L * D * D
        scores = 2 * N * layer.heads * L * L * layer.dh
        weighted = scores
        return proj + scores + weighted, shape
    if isinstance(layer, TransformerBlock):
        total, s = 0, shape
        for sub in (layer.ln1, layer.attn, layer.ln2, layer.fc1, layer.act, layer.fc2):
            f, s = _flops_of(sub, s, N)
            total += f
        return total, s
    if isinstance(layer, PatchEmbedding):
        return _flops_of(layer.conv, shape, N)
    # ---- 自定义容器：有 layers 属性就递归 ----
    if hasattr(layer, 'layers') and isinstance(getattr(layer, 'layers'), list):
        total, s, f = 0, shape, None
        for sub in layer.layers:
            f, s = _flops_of(sub, s, N)
            total += f
        return total, s
    if hasattr(layer, 'blocks') and isinstance(getattr(layer, 'blocks'), list):
        total, s = 0, shape
        for sub in layer.blocks:
            f, s = _flops_of(sub, s, N)
            total += f
        return total, s
    if hasattr(layer, 'stem'):                       # VGG/MobileNet/ResNet 的 stem
        total, s = _flops_of(layer.stem, shape, N)
        for sub in getattr(layer, 'blocks', []):
            f, s = _flops_of(sub, s, N)
            total += f
        for sub in getattr(layer, 'stages', []):
            f, s = _flops_of(sub, s, N)
            total += f
        if hasattr(layer, 'head'):
            f, s = _flops_of(layer.head, s, N)
            total += f
        return total, s
    return 0, shape


def count_params(model):
    """参数量（可训练参数元素个数）。"""
    return int(sum(p.data.size for p in model.params()))


def count_flops(model, input_shape, batch=1):
    """计算模型乘加次数（FLOPs = 2 × MACs），返回 (flops, macs)。

    参数：
        model: 具有 forward 的模型；
        input_shape (tuple): 不含 batch 的输入形状，如 (1, 64, 64)；
        batch (int): batch 大小，默认 1（便于跨框架对比“单样本 FLOPs”）。

    说明：按本文件头部的统一公式计算；分组卷积只算每组实际看到的通道数。
    """
    shape = tuple(int(d) for d in input_shape)
    if hasattr(model, 'layers'):                     # Sequential
        total = 0
        for layer in model.layers:
            f, shape = _flops_of(layer, shape, batch)
            total += f
        return int(total), int(total // 2)
    f, _ = _flops_of(model, shape, batch)
    return int(f), int(f // 2)


def measure_latency(model, input_shape, batch=1, warmup=3, repeat=20):
    """测量单帧/单批推理延迟（CPU，评估模式）。

    返回：
        dict: mean_ms / p95_ms / fps / batch。
    """
    model.train_mode(False)
    x = np.random.randn(batch, *input_shape).astype(np.float64)
    for _ in range(warmup):
        _ = model(Node(x))
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = model(Node(x))
        times.append(time.perf_counter() - t0)
    mean_ms = float(np.mean(times) * 1000)
    return {'mean_ms': mean_ms,
            'p95_ms': float(np.percentile(times, 95) * 1000),
            'fps': float(1000.0 / mean_ms) if mean_ms > 0 else float('inf'),
            'batch': int(batch)}


def _iter_layers(obj, seen=None):
    """递归遍历模型里的所有 Layer 实例（用于统计激活内存/包装 forward）。

    通过检查 __dict__ 中的 Layer 实例与 Layer 列表实现，不依赖具体模型结构，
    对 Sequential / ResNet / VGG / MobileNet / ViT 都适用。
    """
    from miniflows.layers import Layer
    if seen is None:
        seen = set()
    if not isinstance(obj, Layer) or id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    for value in vars(obj).values():
        if isinstance(value, Layer):
            yield from _iter_layers(value, seen)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Layer):
                    yield from _iter_layers(item, seen)


def measure_activation_bytes(model, input_shape, batch=1):
    """估算单次前向的峰值激活内存（字节），不含梯度与反向缓存。

    做法：临时给每个子层的 **实例** forward 套一层记录器（实例属性会遮蔽
    类方法），前向一遍累加每层输出的 nbytes，结束后恢复原 forward。
    这样复合模型（ResNet/VGG/MobileNet/ViT）也能统计到内部各层。

    注意：这是训练内存的下界；反向还要保存中间变量，通常为前向的 2~3 倍。
    """
    from miniflows.layers import Layer
    shape = tuple(int(d) for d in input_shape)
    total = [0]

    def _wrap(layer):
        """包装单层的 forward：调用原实现并累加输出字节数。"""
        original = layer.forward

        def wrapped(x):
            out = original(x)
            data = getattr(out, 'data', None)
            if data is not None:
                total[0] += int(data.nbytes)
            return out
        return wrapped

    patched = []                                  # [(layer, 是否有实例级 forward)]
    for layer in _iter_layers(model):
        had = 'forward' in layer.__dict__         # 记录原来是否有实例属性，便于精确恢复
        patched.append((layer, had, layer.__dict__.get('forward')))
        layer.forward = _wrap(layer)
    try:
        x = Node(np.zeros((batch, *shape)))
        _ = model(x)
    finally:
        for layer, had, old in patched:            # 恢复，避免污染模型
            if had:
                layer.forward = old
            else:
                layer.__dict__.pop('forward', None)
    return int(total[0])


def profile_model(model, input_shape, batch=1, nn_name='model'):
    """一次性返回参数量 / FLOPs / 延迟 / 激活内存，便于写入对比表。

    返回：
        dict: name/params/flops/macs/latency_ms/fps/act_mb/input。
    """
    flops, macs = count_flops(model, input_shape, batch=batch)
    lat = measure_latency(model, input_shape, batch=batch)
    act = measure_activation_bytes(model, input_shape, batch=batch)
    return {
        'name': nn_name,
        'params': count_params(model),
        'flops': flops,
        'macs': macs,
        'flops_human': f'{flops / 1e6:.2f} MFLOPs',
        'latency_ms': lat['mean_ms'],
        'p95_ms': lat['p95_ms'],
        'fps': lat['fps'],
        'act_mb': act / (1024 * 1024),
        'input': f'{batch}×{tuple(input_shape)}',
    }
