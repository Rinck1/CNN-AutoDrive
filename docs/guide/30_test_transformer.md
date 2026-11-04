# `tests/test_transformer.py` 阅读指南

> 前置：transformer.py、autodiff ｜ 难度：★★ ｜ 建议 30 分钟

## 1. 文件定位

ViT/Transformer 组件的单元测试：

1. 新算子（GELU/concat/broadcast_to/select/4D matmul）梯度检验；
2. LayerNorm 统计性质；
3. 极小 ViT 前向、注意力形状、全参数梯度；
4. ViT 保存/加载往返。

## 2. 结构速览

| 顺序 | 测试/行号 | 检查什么 |
|---|---|---|
| 1 | `test_new_ops_gradients` (41) | 5 项梯度检验 |
| 2 | `test_layernorm_stats` (84) | 均值≈0、方差≈1（eps=0） |
| 3 | `test_tiny_vit` (97) | 前向形状、注意力形状、全参数梯度 |
| 4 | `test_save_load_vit` (117) | npz 往返一致 |

## 3. 逐段讲解

### 3.1 新算子

| 算子 | 输入 | 检查 |
|---|---|---|
| gelu | (3,5) | max_rel 4.5e-10 |
| concat | (3,4)+(3,4) axis=1 | 2.9e-11 |
| broadcast_to | (2,1,3)→(4,2,1,3) | 5.6e-11 |
| select | (2,5,3) 取 axis=1 index=2 | 2.7e-11 |
| 4D matmul | (2,3,4,5)@(2,3,5,4) | 6.8e-9 |

全部 <1e-5，证明多头注意力的批量矩阵乘反向正确。

### 3.2 LayerNorm

- 默认 `eps=1e-5` 会让归一化方差略小于 1；
- 测试用 `eps=0` 验证纯归一化：均值 1.4e-16、方差误差 4.4e-16；
- 说明“默认 eps 的偏差是数值保护，不是 bug”。

### 3.3 极小 ViT

```python
build_vit(img_size=16, patch_size=8, dim=16, depth=1, heads=2)
```

- 输入 `(2,1,16,16)` → 输出 `(2,1)`；
- 4 个 patch + 1 CLS = 5 token；
- 注意力形状 `(2,2,5,5)`；
- 全参数 3,409，梯度 max_rel **1.168e-6**。

### 3.4 保存/加载

- 前向一次记录预测；
- `save_model` → 新建模型 → `load_model`；
- 预测最大差必须为 **0.0**。

## 4. 运行

```bash
python tests/test_transformer.py
```

末尾输出 `✓ Transformer 全部测试通过`。

## 5. 易错点

1. LayerNorm 测试必须用 `eps=0`，否则方差误差约 1e-6；
2. maxpool/ReLU 等不可导点不在本测试范围；
3. 极小 ViT 的 `img_size` 必须能被 patch 整除；
4. `attention_maps` 必须在一次前向之后调用；
5. 保存/加载要求参数名与结构完全一致。

## 6. 自检问题

1. 为什么 4D matmul 是 Transformer 的关键？
2. LayerNorm 的 eps 对统计检验有什么影响？
3. 极小 ViT 的参数 3,409 能手算吗？
4. 注意力形状 `(N,H,L+1,L+1)` 各维含义？
