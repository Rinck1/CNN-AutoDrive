# -*- coding: utf-8 -*-
"""
MiniFlows —— 一个纯 numpy 实现的迷你深度学习框架（课程综合设计）
==================================================================

设计目标
--------
1. **简单易用**：接口仿 Keras/PyTorch 的常用写法——
   `model = Sequential(Linear(2, 1))`、`Trainer(model, loss, opt).train(...)`；
2. **原理透明**：计算图、反向传播、卷积、池化、优化器全部手写，
   不依赖任何自动微分库，便于学习与答辩讲解；
3. **可验证**：每个算子都支持数值梯度检验（见 miniflows.autodiff）。

模块导航
--------
======================  ==================================================
模块                     主要内容
======================  ==================================================
miniflows.graph          计算图节点 Node（值、梯度、父节点、反向闭包）
miniflows.ops            算子：加减乘除、矩阵乘、激活、卷积、池化、形状/归约
miniflows.layers         层：Linear/Conv2D/MaxPool2D/.../Sequential 容器
miniflows.losses         MSELoss / BCEWithLogitsLoss / CrossEntropyLoss / L2 正则
miniflows.optim          MBGD（可选动量）/ Adam 优化器
miniflows.autodiff       数值梯度（中心差分）与梯度检验工具
miniflows.data           数据预处理：标准化、one-hot、划分、DataLoader、MNIST/鸢尾花
miniflows.trainer        训练器 Trainer：分批、训练/评估模式、回调、早停
miniflows.utils          准确率、EarlyStopping、学习率调度、模型保存/加载
miniflows.visualization  计算图可视化、损失曲线、通用曲线、图像网格
======================  ==================================================

快速上手
--------
>>> import numpy as np
>>> from miniflows import Sequential, Linear, ReLU, MSELoss, Adam, Trainer
>>> X = np.random.randn(200, 2)
>>> y = X @ np.array([[3.0], [-2.0]]) + 1.0
>>> model = Sequential(Linear(2, 16), ReLU(), Linear(16, 1))
>>> trainer = Trainer(model, MSELoss(), Adam(model.params(), lr=1e-2),
...                   task='regression')
>>> hist = trainer.train(X, y, epochs=50, batch_size=32, verbose=False)

注意：本包**不**在顶层导出 ops 中的 sum/max 等函数，因为它们会遮蔽
Python 内置函数；需要时请显式 `import miniflows.ops as ops` 调用。
"""

# ---------------------------------------------------------------------------
# 计算图：节点与图工具
# ---------------------------------------------------------------------------
from .graph import Node, constant, param, build_topo, collect_edges

# ---------------------------------------------------------------------------
# 层与模型容器（仿 Keras 的 Sequential 写法）
# ---------------------------------------------------------------------------
from .layers import (
    Layer,
    Linear,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    Flatten,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Dropout,
    Sequential,
)

# ---------------------------------------------------------------------------
# Transformer / Vision Transformer（ViT）组件
# ---------------------------------------------------------------------------
from .transformer import (
    LayerNorm,
    GELU,
    MultiHeadSelfAttention,
    TransformerBlock,
    PatchEmbedding,
    ViTRegressor,
    build_vit,
)

# ---------------------------------------------------------------------------
# ResNet 组件
# ---------------------------------------------------------------------------
from .resnet import GroupNorm2D, BasicBlock, ResNetRegressor, build_resnet

# ---------------------------------------------------------------------------
# 损失函数与正则
# ---------------------------------------------------------------------------
from .losses import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss, l2_penalty

# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------
from .optim import Optimizer, MBGD, Adam

# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------
from .trainer import Trainer

# ---------------------------------------------------------------------------
# 工具函数：指标、早停、学习率调度、模型存取
# ---------------------------------------------------------------------------
from .utils import (
    set_seed,
    accuracy,
    mse,
    EarlyStopping,
    LRScheduler,
    save_model,
    load_model,
    count_params,
)

# 版本号：课程验收时便于标识提交状态
__version__ = "1.0.0"

# 显式声明公开 API，避免 `from miniflows import *` 时意外导入内部工具
__all__ = [
    # 计算图
    "Node", "constant", "param", "build_topo", "collect_edges",
    # 层
    "Layer", "Linear", "Conv2D", "MaxPool2D", "AvgPool2D", "Flatten",
    "ReLU", "Sigmoid", "Tanh", "Softmax", "Dropout", "Sequential",
    # Transformer / ViT
    "LayerNorm", "GELU", "MultiHeadSelfAttention", "TransformerBlock",
    "PatchEmbedding", "ViTRegressor", "build_vit",
    # ResNet
    "GroupNorm2D", "BasicBlock", "ResNetRegressor", "build_resnet",
    # 损失
    "MSELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "l2_penalty",
    # 优化器
    "Optimizer", "MBGD", "Adam",
    # 训练器
    "Trainer",
    # 工具
    "set_seed", "accuracy", "mse", "EarlyStopping", "LRScheduler",
    "save_model", "load_model", "count_params",
]
