# `miniflows/transformer.py` 阅读指南

> 前置：graph/ops/layers、注意力机制 ｜ 难度：★★★ ｜ 建议 90 分钟 ｜ 进阶

## 1. 文件定位

在 MiniFlows 上实现 encoder-only Transformer / ViT：

```
PatchEmbedding → [CLS] + 位置编码 → N×TransformerBlock → LayerNorm
→ 取 [CLS] → Linear → Tanh
```

所有反向由既有算子自动完成，没有手写整网反向。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `_prefixed` (40) | 子模块参数名加前缀 |
| 2 | `LayerNorm` (48) | 沿最后一维归一化 |
| 3 | `GELU` (91) | 包装 `ops.gelu` |
| 4 | `MultiHeadSelfAttention` (102) | Q/K/V/O + 多头注意力 |
| 5 | `TransformerBlock` (174) | Pre-LN 残差 + MLP |
| 6 | `PatchEmbedding` (225) | 卷积切 patch |
| 7 | `ViTRegressor` (263) | 完整 ViT 回归器 |
| 8 | `build_vit` (351) | 默认配置工厂 |

## 3. 逐段讲解

### 3.1 LayerNorm（用算子组合，不手写反向）

```python
mu  = mean(x, axis=-1, keepdims=True)
xc  = x - mu
var = mean(xc*xc, axis=-1, keepdims=True)
y   = (xc / sqrt(var + eps)) * gamma + beta
```

- 沿最后一维（特征维）归一化；
- `gamma/beta` 形状 `(D,)`，广播到 `(N,L,D)`；
- 反向由 mean/sub/mul/div/sqrt 自动求出；
- 默认 `eps=1e-5`；单测用 `eps=0` 验证纯归一化。

### 3.2 MultiHeadSelfAttention

```python
Q = Wq(x), K = Wk(x), V = Wv(x)          # 三个 Linear
拆头: (N,L,D) → (N,L,H,Dh) → (N,H,L,Dh)
scores = Q @ Kᵀ / sqrt(Dh)               # (N,H,L,L)
attn   = softmax(scores, axis=-1)
head   = attn @ V                        # (N,H,L,Dh)
合并: transpose → reshape → Wo           # (N,L,D)
```

- 用 4D 批量 `matmul`，不需要手写循环；
- `1/sqrt(Dh)` 防止点积随维度增大；
- `last_attn` 缓存最近一次注意力 `(N,H,L,L)`，供可视化；
- 只有自注意力，**没有 mask、没有 cross-attention**。

### 3.3 TransformerBlock（Pre-LN）

```python
x = x + Attn(LayerNorm(x))
x = x + MLP(LayerNorm(x))
MLP = Linear(D→rD) → GELU → Linear(rD→D)
```

- Pre-LN 让残差主路保持恒等映射，小数据/浅层更稳；
- 每个 block 的 `named_params` 前缀为 `block{i}.`。

### 3.4 PatchEmbedding

```python
y = Conv2D(C→D, k=patch, stride=patch)(x)   # (N,D,H/p,W/p)
y = reshape(y, (N,D,L)) → transpose → (N,L,D)
```

- 等价于每个 patch 展平后过同一个线性层；
- 64×64、patch=8 → 64 个 token。

### 3.5 ViTRegressor

```python
tokens = patch_embed(x)                         # (N,L,D)
cls = broadcast_to(cls_token, (N,1,D))
x = concat([cls, tokens], axis=1) + pos_embed   # (N,L+1,D)
for block in blocks: x = block(x)
x = norm(x)
z = select(x, 0, axis=1)                        # [CLS]
y = tanh(head(z))                               # (N,1)
```

- `cls_token` 初始化为 0，位置编码小方差正态；
- `named_params` 扁平化所有子层参数；
- `attention_maps(index)` 返回第 index 个 block 的注意力。

## 4. 关于注意力图的重要说明

默认 ViT 的注意力**不是稀疏定位**：

- 第 1 个 block 的 CLS query 几乎由位置编码决定（CLS 初始为 0）；
- 第 2 个 block 的注意力接近均匀（归一化熵约 0.9）；
- 跨样本余弦相似度高，说明更像固定位置先验；
- 与道路/转向的相关性接近 0。

因此：

- **注意力机制有用**：消融成均匀权重后 MAE 从 0.045 崩到 0.34；
- **注意力热图不能当解释**：它展示的是原始权重，不是因果重要性；
- 报告里应标注“仅示意”，不要写“模型主要关注这里”。

## 5. 易错点

1. **3D 偏置对齐**：`(N,L,D)` 的 Linear 偏置必须对齐最后一维；
2. **拆头顺序**：reshape 成 `(N,L,H,Dh)` 再 transpose 成 `(N,H,L,Dh)`；
3. **LayerNorm 轴**：必须最后一维，不是 batch 维；
4. **CLS 初始化**：0 初始化导致 block0 注意力不可解释；
5. **8×8 patch**：分辨率粗，2px 边线定位能力有限；
6. **没有 mask**：序列必须定长、矩形。

## 6. 自检问题

1. 写出 MHSA 的完整形状变化；
2. Pre-LN 与 Post-LN 的区别？为什么这里选 Pre-LN？
3. LayerNorm 的 γ/β 如何广播？反向由哪些算子完成？
4. 为什么注意力图不能当解释？用哪两个指标判断？
5. 如何把 `ViTRegressor` 改成分类模型？
