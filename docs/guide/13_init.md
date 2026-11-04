# `miniflows/__init__.py` 阅读指南

> 前置：各模块 ｜ 难度：★ ｜ 建议 15 分钟 ｜ API

## 1. 文件定位

MiniFlows 的**顶层公开 API**。它决定用户能 `from miniflows import ...` 拿到什么，
也定义了模块的导入顺序。

## 2. 内容速览

- 文件头：框架定位、模块导航、快速上手示例；
- 依次导入：`graph → layers → transformer → losses → optim → trainer → utils`；
- `__version__ = "1.0.0"`；
- `__all__` 列出 40 个公开符号。

## 3. 导入顺序为什么重要

```python
from .graph import Node, ...
from .layers import ...
from .transformer import ...
from .losses import ...
```

- `transformer` 依赖 `layers` 与 `ops`，必须在 `layers` 之后导入；
- `ops` 不在顶层导出 `sum/max`，因为它们会遮蔽 Python 内置函数；
  需要时显式 `import miniflows.ops as ops`；
- 循环依赖风险：`transformer.py` 用 `from miniflows import ops`，
  此时 `ops` 已由 `layers` 的导入链加载。

## 4. 公开符号分组

| 分组 | 符号 |
|---|---|
| 图 | `Node constant param build_topo collect_edges` |
| 层 | `Layer Linear Conv2D MaxPool2D AvgPool2D Flatten ReLU Sigmoid Tanh Softmax Dropout Sequential` |
| Transformer | `LayerNorm GELU MultiHeadSelfAttention TransformerBlock PatchEmbedding ViTRegressor build_vit` |
| 损失 | `MSELoss BCEWithLogitsLoss CrossEntropyLoss l2_penalty` |
| 优化 | `Optimizer MBGD Adam` |
| 训练 | `Trainer` |
| 工具 | `set_seed accuracy mse EarlyStopping LRScheduler save_model load_model count_params` |

## 5. 易错点

1. 不要 `from miniflows import *` 后期望拿到 `ops.sum/max`；它们不在 `__all__`；
2. 不要在这里 import torch/sklearn，保持核心只依赖 numpy；
3. 新增模块时同步更新 `__all__`，否则 `from miniflows import *` 看不到；
4. 版本号变化应同步 README/文档。

## 6. 自检问题

1. 为什么顶层不导出 `ops.sum`？
2. `transformer` 的导入为什么必须放在 `layers` 之后？
3. 如何在 REPL 里快速确认 API 是否完整？
