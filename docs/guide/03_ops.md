# `miniflows/ops.py` 阅读指南

> 前置：`graph.py`、矩阵乘法、卷积基础 ｜ 难度：★★★ ｜ 建议 120 分钟 ｜ 必读

## 1. 文件定位

全部**算子**的实现：基础运算、激活、卷积、池化、形状/归约、ViT 新增算子。
每个算子遵循同一模式：

```
输入 Node（或自动包装） → 算前向值 → 新建 Node → 注册 _backward 闭包 → 返回
```

反向闭包只写“局部雅可比 × 上游梯度”，链式法则由 `Node.backward()` 统一调度。

## 2. 结构速览

| 区块 | 行号 | 内容 |
|---|---|---|
| 内部工具 | 46~204 | `_as_node`、`_bind`、`unbroadcast`、`im2col/col2im`、池化窗口/散射 |
| 基础运算 | 206~346 | `add/sub/mul/div/matmul/neg` |
| 激活 | 360~500 | `exp/log/pow/sqrt/relu/sigmoid/tanh/softmax` |
| ViT 新增 | 501~625 | `gelu/concat/broadcast_to/select` |
| 卷积池化 | 626~700 | `conv2d/maxpool2d/avgpool2d` |
| 形状/归约 | 700+ | `reshape/flatten/transpose/sum/mean/max` |

## 3. 逐段讲解

### 3.1 三个内部工具（必须懂）

**`_as_node(x)`**：Node 原样返回；ndarray/标量包成常量 Node。

**`_bind(out, parents, op_name)`**：设置 `out.parents`、`out.op_name`，
统一 requires_grad 传播（任一父节点需要梯度，输出就需要）。

**`unbroadcast(grad, shape)`**：这是广播正确性的核心。
若前向把 `(D,)` 广播到 `(N,L,D)`，反向要把梯度沿广播轴求和回 `(D,)`。
实现：把 grad reshape 成目标形状（左侧补 1），再对多出来的轴 `sum`。

### 3.2 基础运算

| 算子 | 前向 | 反向 |
|---|---|---|
| `add` | `a+b` | `da+=g`, `db+=g`（unbroadcast） |
| `sub` | `a-b` | `da+=g`, `db+=-g` |
| `mul` | `a*b` | `da+=g*b`, `db+=g*a` |
| `div` | `a/b` | `da+=g/b`, `db+=-a*g/b²` |
| `matmul` | `a@b` | `da=g@bᵀ`, `db=aᵀ@g`，用 `swapaxes(-1,-2)` 支持批量维 |
| `neg` | `-x` | `-g` |

`matmul` 的批量维写法是 Transformer 能跑的关键：
`(N,H,L,Dh) @ (N,H,Dh,L)` 会自动在最后两维做矩阵乘。

### 3.3 激活函数

- `relu`：`max(0,x)`，反向 `(x>0)*g`；0 点约定导数为 0；
- `sigmoid`：数值稳定写法，反向 `y(1-y)g`；
- `tanh`：`np.tanh`，反向 `(1-y²)g`；
- `softmax`：减最大值稳定化；反向用雅可比对角化技巧
  `dz = y*(g - Σ(g*y))`，避免构造 L×L 雅可比；
- `gelu`：tanh 近似，反向公式见代码注释，用于 Transformer。

### 3.4 卷积（重点）

前向：

1. 对输入做 padding；
2. `im2col`：每个 `(C,KH,KW)` 感受野展平成一行 → `(N·OH·OW, C·KH·KW)`；
3. 权重展平 `(C·KH·KW, F)`；
4. `out = col @ Wmat + b`，reshape 回 `(N,F,OH,OW)`。

反向：

- `dWmat = colᵀ @ g_flat`，`db = g` 在 `(N,OH,OW)` 上求和；
- `dcol = g_flat @ Wmatᵀ`；
- `col2im` 把每行散射回补零后的 `dx`，重叠位置 `+=` 累加；
- 去掉 padding 区域。

形状公式：`OH = (H + 2p - KH)//s + 1`。

### 3.5 池化

- `_pool_windows` 用切片取每个窗口；
- `maxpool2d` 前向记录 argmax（或反向重算），反向只把梯度放到最大值位置；
- `avgpool2d` 反向把 `g/(kh*kw)` 均分到窗口每个位置。

### 3.6 形状/归约

- `reshape/flatten/transpose` 是纯视图变换，反向按逆变换还原；
- `sum/mean` 反向把梯度广播回原形状；`_expand_reduced_grad` 负责补轴；
- `max(x)` 全量最大值，梯度只流向最大值位置；
- 注意：本模块定义了 `sum/max`，会遮蔽 Python 内置函数；
  内部实现里不能用 `max(a,b)`，这也是曾经的一个真 bug。

### 3.7 ViT 新增算子

| 算子 | 前向 | 反向 |
|---|---|---|
| `concat` | 沿 axis 拼接 | 按长度 `np.split` 梯度 |
| `broadcast_to` | 广播到目标形状 | `unbroadcast` 求和 |
| `select` | 取 axis 的第 i 个切片 | 零梯度 + 散射回切片 |
| `gelu` | tanh 近似 | 解析导数 |

`broadcast_add` 的偏置对齐规则：

- `a` 是 4D 卷积特征图 `(N,C,H,W)`：`(F,)` 补成 `(1,F,1,1)`；
- `a` 是 3D 序列 `(N,L,D)`：`(F,)` 补成 `(1,1,F)`（最后一维）。

这条规则是 ViT 接入时补的；写死“按通道维”会让 Linear 在 3D 输入上直接报错。

## 4. 调用关系

`layers/losses/transformer` 都只通过本模块的算子建图；
`autodiff` 用本模块算数值梯度；`visualization` 只读 Node 的 op_name/parents。

## 5. 易错点

1. **广播梯度**：`unbroadcast` 必须沿“被扩张的轴”求和，不能简单 reshape；
2. **卷积反向重叠累加**：`col2im` 里必须是 `+=`；
3. **maxpool 并列最大值**：按次梯度约定分给所有并列位置；
4. **softmax 轴**：损失内部和注意力都用 axis=-1，传错轴会静默错；
5. **sqrt/log 定义域**：输入 0/负数会产生 nan，损失里需加 eps 或用稳定公式；
6. **内置函数遮蔽**：`sum/max` 是算子函数，内部代码避免直接调用内置同名函数。

## 6. 自检问题

1. 手推 `matmul` 的 `dA/dB`，并解释批量维为什么用 `swapaxes(-1,-2)`；
2. `unbroadcast((N,L,D)→(D,))` 具体做了哪些 sum？
3. im2col 的列矩阵形状是什么？`col2im` 为什么需要累加？
4. softmax 反向为什么不用构造雅可比矩阵？
5. `broadcast_add` 对 4D 与 3D 的对齐规则分别是什么？
