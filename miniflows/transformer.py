# -*- coding: utf-8 -*-
"""
miniflows/transformer.py —— 基于 MiniFlows 的 Vision Transformer（ViT）组件
============================================================================

【为什么单独一个模块】
    CNN 的归纳偏置（局部性、平移等变）由卷积核提供；ViT 则把图像切成
    固定大小的 patch，当作"视觉词序列"送入标准 Transformer 编码器。
    本模块在 MiniFlows 的自动微分之上，用 **运算符组合** 实现：

        PatchEmbedding → [CLS] + 位置编码 → N × TransformerBlock
        → LayerNorm → 取 [CLS] → Linear → Tanh

【实现要点】
    1. 全部梯度由 MiniFlows 的 Node/ops 自动求导，不手写整网反向；
    2. LayerNorm 直接用 mean/sub/mul/div 组合实现，反向自动正确；
    3. 多头注意力用批量 matmul（(N,H,L,Dh)），softmax 沿最后一维；
    4. 用 [CLS] token 的最终表示做回归（也可改为 patch 平均池化）；
    5. 每个 block 都是 Pre-LN 残差结构：x = x + Attn(LN(x))，
       x = x + MLP(LN(x))，训练更稳定。

【参数量（64×64、patch=8、dim=64、depth=3、heads=4）】
    patch conv 4,096 + 位置编码 4,160 + CLS 64
    + 3 × block ≈ 100,608 + 最终 LN 128 + 回归头 65 ≈ 109,121
    （具体数值由 count_params 打印，因初始化/结构参数略有差异）

【依赖】numpy + miniflows.graph/ops/layers（无循环依赖）。
"""

import numpy as np

from miniflows.graph import Node, param
from miniflows.layers import Layer, Linear, Conv2D, Tanh
from miniflows import ops


# ======================================================================
# 参数命名小工具
# ======================================================================
def _prefixed(module, prefix):
    """把子模块的 named_params 加上 'prefix.' 前缀，供嵌套层使用。"""
    return [(f'{prefix}.{name}', p) for name, p in module.named_params()]


# ======================================================================
# 一、LayerNorm
# ======================================================================
class LayerNorm(Layer):
    """层归一化：对最后一维（特征维）做零均值、单位方差归一化。

    公式：
        μ = mean(x, -1, keepdim)
        σ² = mean((x-μ)², -1, keepdim)
        y = γ · (x-μ) / √(σ²+ε) + β

    实现说明：
        不手写反向，直接用 ops.mean/sub/mul/div/sqrt 组合，
        梯度由计算图自动求出——这也顺带验证了这些基础算子的正确性。
        γ、β 为可训练参数，形状 (D,)，广播到 (N,L,D)。
    """

    def __init__(self, dim, eps=1e-5, name=None):
        """参数：dim 特征维；eps 防除零小量；name 层名。"""
        super().__init__(name)
        self.dim = int(dim)
        self.eps = float(eps)
        self.gamma = param(np.ones(dim, dtype=np.float64), name='gamma')   # 缩放
        self.beta = param(np.zeros(dim, dtype=np.float64), name='beta')    # 平移

    def forward(self, x):
        """x:(..., D) -> 同形状归一化结果。"""
        mu = ops.mean(x, axis=-1, keepdims=True)          # (...,1)
        xc = ops.sub(x, mu)                               # 零均值
        var = ops.mean(ops.mul(xc, xc), axis=-1, keepdims=True)  # 方差
        std = ops.sqrt(ops.add(var, self.eps))            # √(σ²+ε)
        y = ops.div(xc, std)                              # 标准化
        return ops.add(ops.mul(y, self.gamma), self.beta)  # γ·y+β（广播）

    def named_params(self):
        """返回 [('gamma', Node), ('beta', Node)]。"""
        return [('gamma', self.gamma), ('beta', self.beta)]

    def params(self):
        """返回全部可训练参数。"""
        return [self.gamma, self.beta]


# ======================================================================
# 二、GELU 激活层
# ======================================================================
class GELU(Layer):
    """GELU 激活（tanh 近似），直接复用 ops.gelu 的反向闭包。"""

    def forward(self, x):
        """x -> GELU(x)，形状不变。"""
        return ops.gelu(x)


# ======================================================================
# 三、多头自注意力
# ======================================================================
class MultiHeadSelfAttention(Layer):
    """多头自注意力（MHSA）：把 D 维特征拆成 H 个头，每头维度 Dh=D/H。

    前向：
        Q,K,V = xWq+bq, xWk+bk, xWv+bv          （三个独立 Linear）
        每个头：scores = QKᵀ/√Dh → softmax → 加权求和 V
        拼接所有头 → 输出投影 Wo

    形状（N 批、L 序列长度、D 特征维）：
        x      : (N, L, D)
        Q/K/V  : (N, H, L, Dh)      （把 D 维 reshape 后转置）
        scores : (N, H, L, L)
        attn   : (N, H, L, L)       （最后一维归一化）
        输出    : (N, L, D)

    反向：完全由 ops.matmul/softmax/reshape/transpose 自动完成。
    """

    def __init__(self, dim, heads, name=None):
        """参数：dim 特征维；heads 头数（必须整除 dim）；name 层名。"""
        super().__init__(name)
        if dim % heads != 0:
            raise ValueError(f'dim={dim} 必须能被 heads={heads} 整除')
        self.dim = int(dim)
        self.heads = int(heads)
        self.dh = dim // heads
        # 四个线性投影：Q、K、V、输出；偏置由 Linear 自带
        self.Wq = Linear(dim, dim, name='q')
        self.Wk = Linear(dim, dim, name='k')
        self.Wv = Linear(dim, dim, name='v')
        self.Wo = Linear(dim, dim, name='o')
        self.last_attn = None        # 缓存最近一次注意力权重，供可视化

    def forward(self, x):
        """x:(N,L,D) -> (N,L,D)。"""
        N, L, D = x.data.shape
        H, Dh = self.heads, self.dh

        # ---- 1) 线性投影，并拆头：(N,L,D) -> (N,L,H,Dh) -> (N,H,L,Dh) ----
        q = ops.transpose(ops.reshape(self.Wq(x), (N, L, H, Dh)), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(self.Wk(x), (N, L, H, Dh)), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(self.Wv(x), (N, L, H, Dh)), (0, 2, 1, 3))

        # ---- 2) 缩放点积注意力 ----
        # scores = Q @ Kᵀ / √Dh；Kᵀ 只需交换最后两维
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = ops.mul(scores, 1.0 / np.sqrt(Dh))       # 防止点积随 Dh 增大
        attn = ops.softmax(scores, axis=-1)               # 每个 query 行归一化
        self.last_attn = np.array(attn.data, copy=True)   # 供注意力图可视化

        # ---- 3) 加权求和并合并头 ----
        y = ops.matmul(attn, v)                            # (N,H,L,Dh)
        y = ops.transpose(y, (0, 2, 1, 3))                 # (N,L,H,Dh)
        y = ops.reshape(y, (N, L, D))                      # (N,L,D)
        return self.Wo(y)                                  # 输出投影

    def named_params(self):
        """q/k/v/o 四组参数的扁平命名。"""
        out = []
        for tag, layer in [('q', self.Wq), ('k', self.Wk),
                           ('v', self.Wv), ('o', self.Wo)]:
            out.extend(_prefixed(layer, tag))
        return out

    def params(self):
        """全部可训练参数。"""
        return [p for _, p in self.named_params()]


# ======================================================================
# 四、Transformer Block（Pre-LN + 残差）
# ======================================================================
class TransformerBlock(Layer):
    """一个 Transformer 编码器块：

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
        MLP = Linear(D→rD) → GELU → Linear(rD→D)

    采用 Pre-LN（先归一化再进子层）而不是原始 ViT 的 Post-LN：
    残差主路上没有归一化，深层梯度更稳定，小数据上更容易训练。
    """

    def __init__(self, dim, heads, mlp_ratio=2, name=None):
        """参数：dim 特征维；heads 头数；mlp_ratio 隐层扩张倍数；name 层名。"""
        super().__init__(name)
        hidden = int(dim * mlp_ratio)
        self.ln1 = LayerNorm(dim, name='ln1')
        self.attn = MultiHeadSelfAttention(dim, heads, name='attn')
        self.ln2 = LayerNorm(dim, name='ln2')
        self.fc1 = Linear(dim, hidden, name='fc1')
        self.act = GELU()
        self.fc2 = Linear(hidden, dim, name='fc2')

    def forward(self, x):
        """x:(N,L,D) -> 同形状；两次残差连接。"""
        x = ops.add(x, self.attn(self.ln1(x)))          # 残差 1：注意力
        x = ops.add(x, self.fc2(self.act(self.fc1(self.ln2(x)))))  # 残差 2：MLP
        return x

    def named_params(self):
        """ln1 / attn / ln2 / fc1 / fc2 的参数名。"""
        out = []
        out.extend(_prefixed(self.ln1, 'ln1'))
        out.extend(_prefixed(self.attn, 'attn'))
        out.extend(_prefixed(self.ln2, 'ln2'))
        out.extend(_prefixed(self.fc1, 'fc1'))
        out.extend(_prefixed(self.fc2, 'fc2'))
        return out

    def params(self):
        """全部可训练参数。"""
        return [p for _, p in self.named_params()]

    def train_mode(self, flag):
        """递归切换子层训练/评估模式（当前只有 Dropout 会用到）。"""
        for layer in (self.ln1, self.attn, self.ln2, self.fc1, self.fc2):
            layer.train_mode(flag)


# ======================================================================
# 五、Patch Embedding
# ======================================================================
class PatchEmbedding(Layer):
    """把图像切成 patch 并映射为 D 维向量。

    用 Conv2D(kernel=patch, stride=patch, padding=0) 等价于：
        每个 patch 展平后过同一个线性层（权重共享），
    输出 (N,D,H/p,W/p) → reshape/transpose → (N,L,D)，L=(H/p)(W/p)。

    例：64×64、patch=8 → 输出 (N,64,8,8) → 序列 (N,64,64)。
    """

    def __init__(self, in_channels=1, patch_size=8, dim=64, name=None):
        """参数：输入通道、patch 边长、嵌入维度、层名。"""
        super().__init__(name)
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        self.conv = Conv2D(in_channels, dim, kernel_size=patch_size,
                           stride=patch_size, padding=0, name='conv')

    def forward(self, x):
        """x:(N,C,H,W) -> (N,L,D)。"""
        y = self.conv(x)                                  # (N,D,H/p,W/p)
        N, D, Hp, Wp = y.data.shape
        y = ops.reshape(y, (N, D, Hp * Wp))               # (N,D,L)
        y = ops.transpose(y, (0, 2, 1))                   # (N,L,D)
        return y

    def named_params(self):
        """patch 卷积的 W/b。"""
        return _prefixed(self.conv, 'conv')

    def params(self):
        """全部可训练参数。"""
        return [p for _, p in self.named_params()]


# ======================================================================
# 六、ViT 回归器（DonkeyCar 转向角）
# ======================================================================
class ViTRegressor(Layer):
    """用于转向角回归的 ViT：

        patches = PatchEmbed(x)                       # (N,L,D)
        x = concat([CLS], patches) + pos_embed        # (N,L+1,D)
        x = Block_1(x) ... Block_depth(x)
        x = LayerNorm(x)
        z = x[:, 0]                                   # 取 [CLS]
        y = Tanh(Linear(z))                           # 转向角 ∈ [-1,1]

    说明：使用可学习的 [CLS] token 汇聚全局信息，这是标准 ViT 做法；
    如果输入尺寸改变，L 会变化，位置编码需相应重建。
    """

    def __init__(self, img_size=64, patch_size=8, dim=64, depth=3,
                 heads=4, mlp_ratio=2, in_channels=1, name='vit'):
        """参数：图像边长、patch 边长、嵌入维、block 数、头数、MLP 扩张比。"""
        super().__init__(name)
        if img_size % patch_size != 0:
            raise ValueError('img_size 必须能被 patch_size 整除')
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        self.depth = int(depth)
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim, name='patch_embed')
        # [CLS] token：可学习向量，初始化为 0（常见做法）
        self.cls_token = param(np.zeros((1, 1, dim), dtype=np.float64), name='cls')
        # 位置编码：(1, L+1, D)；小方差正态，避免一开始就压过 patch 信息
        self.pos_embed = param(
            np.random.RandomState(42).normal(0.0, 0.02, (1, self.num_patches + 1, dim)),
            name='pos')
        # 堆叠 Transformer Block
        self.blocks = [TransformerBlock(dim, heads, mlp_ratio, name=f'block{i}')
                       for i in range(depth)]
        self.norm = LayerNorm(dim, name='norm')          # 最终归一化
        self.head = Linear(dim, 1, name='head')          # 回归头
        self.act = Tanh()                                # 输出限制到 [-1,1]

    def forward(self, x):
        """x:(N,1,H,W) -> (N,1) 转向角。"""
        N = x.data.shape[0]
        tokens = self.patch_embed(x)                     # (N,L,D)
        cls = ops.broadcast_to(self.cls_token, (N, 1, self.dim))  # (N,1,D)
        x = ops.concat([cls, tokens], axis=1)            # (N,L+1,D)
        x = ops.add(x, self.pos_embed)                   # 广播加位置编码
        for block in self.blocks:                        # 逐块编码
            x = block(x)
        x = self.norm(x)
        x = ops.select(x, 0, axis=1)                     # 取 [CLS] -> (N,D)
        return self.act(self.head(x))                    # (N,1) ∈ [-1,1]

    def named_params(self):
        """扁平化参数名（含每个 block 的前缀）。"""
        out = []
        out.extend(_prefixed(self.patch_embed, 'patch_embed'))
        out.append(('cls', self.cls_token))
        out.append(('pos', self.pos_embed))
        for i, block in enumerate(self.blocks):
            out.extend(_prefixed(block, f'block{i}'))
        out.extend(_prefixed(self.norm, 'norm'))
        out.extend(_prefixed(self.head, 'head'))
        return out

    def params(self):
        """全部可训练参数。"""
        return [p for _, p in self.named_params()]

    def train_mode(self, flag):
        """递归切换训练/评估模式。"""
        self.patch_embed.train_mode(flag)
        for block in self.blocks:
            block.train_mode(flag)
        self.norm.train_mode(flag)
        self.head.train_mode(flag)

    def attention_maps(self, index=0):
        """返回第 index 个 block 的注意力权重（numpy，(N,H,L,L)）。

        注意：必须在一次前向之后调用（last_attn 在 forward 中缓存）。
        可视化时通常取平均头、取 CLS 行、去掉 CLS 列，得到 L 个 patch 的权重。
        """
        if not 0 <= index < len(self.blocks):
            raise IndexError(f'block 索引越界: {index}')
        return self.blocks[index].attn.last_attn


def build_vit(img_size=64, patch_size=8, dim=64, depth=3, heads=4,
              mlp_ratio=2, in_channels=1):
    """构造 DonkeyCar 用的 ViT 回归模型（默认配置约 10.9 万参数）。"""
    return ViTRegressor(img_size=img_size, patch_size=patch_size, dim=dim,
                        depth=depth, heads=heads, mlp_ratio=mlp_ratio,
                        in_channels=in_channels)
