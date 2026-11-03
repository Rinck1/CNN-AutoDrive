# -*- coding: utf-8 -*-
"""
miniflows/ops.py —— MiniFlows 算子库（函数式接口，闭包注册 _backward）
======================================================================

【模块定位】
    本模块实现框架的全部算子：基础运算 / 激活函数 / 卷积池化 / 形状归约。
    每个算子函数：输入 Node（或自动包装的数组/标量），返回新 Node；
    并在返回的 Node 上注册 ``_backward`` 闭包 —— 闭包内写死了该算子的
    局部求导规则（链式法则的"局部雅可比 × 上游梯度"）。

【各算子反向传播公式一览（x 为输入，g = dL/dy 为上游梯度）】
    add:      y = a + b       dL/da += g, dL/db += g          （广播轴求和还原）
    sub:      y = a - b       dL/da += g, dL/db += -g
    mul:      y = a * b       dL/da += b·g, dL/db += a·g
    div:      y = a / b       dL/da += g/b, dL/db += -a·g/b²
    matmul:   y = a @ b       dL/da += g @ bᵀ, dL/db += aᵀ @ g
    neg:      y = -x          dL/dx += -g
    exp:      y = eˣ          dL/dx += eˣ·g = y·g
    log:      y = ln x        dL/dx += g / x
    pow:      y = xⁿ          dL/dx += n·xⁿ⁻¹·g
    sqrt:     y = √x          dL/dx += g / (2√x) = g / (2y)
    relu:     y = max(0,x)    dL/dx += (x > 0)·g
    sigmoid:  y = 1/(1+e⁻ˣ)   dL/dx += y(1-y)·g
    tanh:     y = tanh(x)     dL/dx += (1 - y²)·g
    gelu:     y = 0.5x(1+tanh(k(x+0.044715x³)))  （tanh 近似）
    softmax:  y = e^{z-max}/Σe^{z-max}    dL/dz += y·(g - Σ(g·y))  （雅可比对角化技巧）
    conv2d:   im2col 展开成矩阵乘，反向 dW = colᵀ·g、dx = col2im(g·Wᵀ)
    maxpool:  反向只把梯度放到前向最大值位置（记录 argmax）
    avgpool:  反向把梯度平均分配到窗口每个位置
    reshape/flatten/transpose: 纯形状变换，反向把梯度按原形状/逆置换还原
    sum:      y = Σx          dL/dx += 广播(g)
    mean:     y = Σx/n        dL/dx += 广播(g)/n

【依赖】numpy + miniflows.graph（单向依赖，无循环）。
"""

import numpy as np

from miniflows.graph import Node, constant


# ============================================================================
#                              内部小工具
# ============================================================================
def _as_node(x):
    """把输入统一为 Node：Node 原样返回，标量/ndarray 自动包装为常量。

    参数：
        x: Node | 数字 | np.ndarray | list
    返回：
        Node
    """
    if isinstance(x, Node):
        return x
    return constant(x)


def _bind(out, parents, op_name):
    """绑定节点的父节点与算子名（让所有算子函数少写两行样板代码）。"""
    out.parents = list(parents)
    out.op_name = op_name


def unbroadcast(grad, shape):
    """把广播后的梯度 grad 沿广播轴求和，还原成目标形状 shape。

    参数：
        grad (np.ndarray): 上游梯度（形状是广播后的结果形状）；
        shape (tuple): 目标输入的原始形状。

    返回：
        np.ndarray：形状恰为 shape 的梯度。

    实现思路：
        广播有两种：① 前导维度补齐（(3,)->(2,3)）—— 对新增的前导轴求和；
        ② size=1 维复制（(1,3)->(2,3)）—— 对该轴求和（keepdims）。
        逐元素广播的偏导数是"每个复制副本各占 1 份"，因此求和即还原。
    """
    shape = tuple(shape)
    if grad.shape == shape:
        return grad                       # 无广播，直接返回
    # ① 梯度比目标多出若干前导维度：逐个对 axis=0 求和压掉
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # ② 目标中 size==1 的维度被广播复制过：沿该轴求和还原（保持维度数）
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ============================================================================
#                        im2col / col2im（卷积加速核心）
# ============================================================================
def im2col(x, kh, kw, stride, padding):
    """把卷积输入按滑动窗口展开成大矩阵（im2col：image to column）。

    参数：
        x (np.ndarray): 输入 (N, C, H, W)；
        kh, kw (int): 卷积核高/宽；
        stride (int): 步幅；
        padding (int): 零填充圈数。

    返回：
        np.ndarray: (N*OH*OW, C*KH*KW) —— 每行是一个感受野窗口摊平后的向量。

    实现思路：
        对核内每个偏移 (i,j)，按步幅 strided-slice 抽出所有窗口对应位置，
        拼成 (KH*KW, N, C, OH, OW) 后重排成 (N*OH*OW, C*KH*KW)。
        这样卷积就变成一个大矩阵乘 col @ W_col（交给 numpy 的 BLAS，
        比六重循环快 1~2 个数量级，是 MNIST 训练的性能关键）。
    """
    x = np.asarray(x, dtype=np.float64)
    N, C, H, W = x.shape
    # 输出尺寸公式：OH = (H + 2p - kh) / stride + 1（整除向下取整）
    OH = (H + 2 * padding - kh) // stride + 1
    OW = (W + 2 * padding - kw) // stride + 1
    if padding > 0:
        # 四维 pad：只对 H/W 两维两侧各补 padding 圈零
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    # 容器：(C, KH, KW, OH, OW) 每个槽位装"所有窗口在该槽位的取值"
    col = np.zeros((N, C, kh, kw, OH, OW), dtype=np.float64)
    for i in range(kh):                 # 核内行偏移
        i_max = i + stride * OH         # 该偏移下 strided-slice 的右端点
        for j in range(kw):             # 核内列偏移
            j_max = j + stride * OW
            # 关键一行：一次性取出所有窗口在 (i,j) 槽位的值
            col[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]
    # 重排成 (N, OH, OW, C, KH, KW) 再摊平成 (N*OH*OW, C*KH*KW)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
    return col


def col2im(col, x_shape, kh, kw, stride, padding):
    """im2col 的伴随运算（adjoint）：把列矩阵散射回图像形状（用于 dx）。

    参数：
        col (np.ndarray): (N*OH*OW, C*KH*KW) 的梯度矩阵；
        x_shape (tuple): 原输入形状 (N, C, H, W)；
        kh, kw, stride, padding: 同 im2col。

    返回：
        np.ndarray: (N, C, H, W) —— 输入 x 的梯度。

    实现思路：
        前向 im2col 是"取值"（多对一读），反向就是"散射累加"（一对多写）：
        重叠窗口的同一像素会被多个窗口共享，故用 += 累加而不是赋值。
        等价于"转置卷积"的散射实现。
    """
    N, C, H, W = x_shape
    OH = (H + 2 * padding - kh) // stride + 1
    OW = (W + 2 * padding - kw) // stride + 1
    # 先还原成 (N, OH, OW, C, KH, KW) 再重排成 (N, C, KH, KW, OH, OW)
    col = col.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    # 多留 stride-1 的余量，避免 strided-slice 越界（经典 fishbook 写法）
    img = np.zeros((N, C, H + 2 * padding + stride - 1,
                    W + 2 * padding + stride - 1), dtype=np.float64)
    for i in range(kh):
        i_max = i + stride * OH
        for j in range(kw):
            j_max = j + stride * OW
            # += ：重叠窗口的梯度在共享像素处累加（与 im2col 的多对一读对偶）
            img[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]
    # 裁掉零填充，只保留有效区域
    return img[:, :, padding:H + padding, padding:W + padding]


# ============================================================================
#                            池化窗口小工具（私有）
# ============================================================================
def _pool_windows(x, kh, kw, stride):
    """抽取池化窗口栈：返回 (K,N,C,OH,OW) 的窗口栈与输出尺寸（K=kh*kw）。"""
    N, C, H, W = x.shape
    OH = (H - kh) // stride + 1          # 池化无 padding（契约约定）
    OW = (W - kw) // stride + 1
    cols = []
    for i in range(kh):
        for j in range(kw):
            # 与 im2col 同款 strided-slice：一次取出所有窗口在 (i,j) 偏移的值
            cols.append(x[:, :, i:i + stride * OH:stride,
                          j:j + stride * OW:stride])
    stack = np.stack(cols, axis=0)       # (KH*KW, N, C, OH, OW)
    return stack, OH, OW


def _pool_scatter(dstack, x_shape, kh, kw, stride):
    """池化窗口栈的散射累加（_pool_windows 的伴随），返回 (N,C,H,W) 梯度。"""
    N, C, H, W = x_shape
    OH = (H - kh) // stride + 1
    OW = (W - kw) // stride + 1
    dx = np.zeros(x_shape, dtype=np.float64)
    k = 0
    for i in range(kh):
        for j in range(kw):
            # 把第 k 个核偏移位置的梯度累加回原输入（重叠窗口用 += 不丢梯度）
            dx[:, :, i:i + stride * OH:stride,
               j:j + stride * OW:stride] += dstack[k]
            k += 1
    return dx


# ============================================================================
#                                基础运算
# ============================================================================
def add(a, b):
    """逐元素加 y = a + b（支持广播）。

    反向：dL/da += g（广播轴求和还原），dL/db += g。
    """
    a, b = _as_node(a), _as_node(b)
    out = Node(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
    _bind(out, [a, b], 'Add')

    def _backward():
        g = out.grad
        if a.requires_grad:
            a.grad += unbroadcast(g, a.data.shape)   # ∂(a+b)/∂a = 1
        if b.requires_grad:
            b.grad += unbroadcast(g, b.data.shape)   # ∂(a+b)/∂b = 1
    out._backward = _backward
    return out


def _broadcast_add_impl(a, b, op_name):
    """broadcast_add/add 共用的偏置广播加实现。"""
    a, b = _as_node(a), _as_node(b)
    out = Node(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
    _bind(out, [a, b], op_name)

    def _backward():
        g = out.grad
        if a.requires_grad:
            a.grad += unbroadcast(g, a.data.shape)
        if b.requires_grad:
            # 偏置形状如 (F,)/(1,F)/(1,F,1,1)：梯度沿所有广播 batch 轴求和还原
            b.grad += unbroadcast(g, b.data.shape)
    out._backward = _backward
    return out


def broadcast_add(a, b):
    """偏置专用广播加：b 形状为 (F,) / (1,F) / (1,F,1,1)，自动广播到 a。

    反向与 add 相同：对 b 的梯度沿所有广播轴求和（unbroadcast 完成）。

    (F,) 偏置的对齐规则（这是 numpy 原生广播做不到的神经网络语义）：
        a 为 4D 卷积特征图 (N,C,H,W) -> b 补成 (1,F,1,1)，与通道维对齐；
        a 为 3D 序列 (N,L,D) 或更高维 -> b 补成 (1,...,1,F)，与最后一维对齐。
    第 2 条是 Transformer 的 Linear 必需的：序列特征是 (N,L,D)，
    偏置必须对齐 D，而不能对齐中间的 L。
    """
    a, b = _as_node(a), _as_node(b)
    if b.data.ndim == 1 and a.data.ndim > 2:
        target_shape = [1] * a.data.ndim
        if a.data.ndim == 4:                   # 卷积特征图：通道维（第 1 维）
            target_shape[1] = b.data.shape[0]
        else:                                  # 序列/高维张量：最后一维
            target_shape[-1] = b.data.shape[0]
        b = reshape(b, tuple(target_shape))     # reshape 自带反向，偏置梯度可回传
    return _broadcast_add_impl(a, b, 'BroadcastAdd')


def sub(a, b):
    """逐元素减 y = a - b。反向：dL/da += g，dL/db += -g。"""
    a, b = _as_node(a), _as_node(b)
    out = Node(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)
    _bind(out, [a, b], 'Sub')

    def _backward():
        g = out.grad
        if a.requires_grad:
            a.grad += unbroadcast(g, a.data.shape)    # ∂(a-b)/∂a = +1
        if b.requires_grad:
            b.grad += unbroadcast(-g, b.data.shape)   # ∂(a-b)/∂b = -1
    out._backward = _backward
    return out


def mul(a, b):
    """逐元素乘 y = a * b。反向：dL/da += b·g，dL/db += a·g。"""
    a, b = _as_node(a), _as_node(b)
    out = Node(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
    _bind(out, [a, b], 'Mul')

    def _backward():
        g = out.grad
        if a.requires_grad:
            a.grad += unbroadcast(b.data * g, a.data.shape)   # ∂(ab)/∂a = b
        if b.requires_grad:
            b.grad += unbroadcast(a.data * g, b.data.shape)   # ∂(ab)/∂b = a
    out._backward = _backward
    return out


def div(a, b):
    """逐元素除 y = a / b。反向：dL/da += g/b，dL/db += -a·g/b²。"""
    a, b = _as_node(a), _as_node(b)
    out = Node(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)
    _bind(out, [a, b], 'Div')

    def _backward():
        g = out.grad
        if a.requires_grad:
            a.grad += unbroadcast(g / b.data, a.data.shape)        # ∂(a/b)/∂a = 1/b
        if b.requires_grad:
            # ∂(a/b)/∂b = -a/b²（商法则）
            b.grad += unbroadcast(-a.data * g / (b.data * b.data), b.data.shape)
    out._backward = _backward
    return out


def matmul(a, b):
    """矩阵乘 y = a @ b（支持 1-D 向量与批量维广播）。

    反向推导（以 2-D 为例，y=a@b）：
        dL/da = g @ bᵀ ；dL/db = aᵀ @ g
    1-D / 批量维的特殊情形也一并处理，多余广播批量维用 unbroadcast 求和还原。
    """
    a, b = _as_node(a), _as_node(b)
    out = Node(np.matmul(a.data, b.data),
               requires_grad=a.requires_grad or b.requires_grad)
    _bind(out, [a, b], 'MatMul')

    def _backward():
        g = out.grad
        if a.requires_grad:
            if b.data.ndim > 1:
                # da = g @ bᵀ（对每个 batch 维同时成立）
                da = np.matmul(g, b.data.swapaxes(-1, -2))
            else:
                # b 是一维：y_i = Σ_k a_ik b_k -> da_ik = g_i b_k（外积广播）
                da = g * b.data
            a.grad += unbroadcast(da, a.data.shape)
        if b.requires_grad:
            if a.data.ndim > 1:
                # db = aᵀ @ g
                db = np.matmul(a.data.swapaxes(-1, -2), g)
            else:
                # a 是一维：y_m = Σ_k a_k b_km -> db_km = a_k g_m（外积广播）
                db = a.data * g
            b.grad += unbroadcast(db, b.data.shape)
    out._backward = _backward
    return out


def neg(x):
    """逐元素取负 y = -x。反向：dL/dx += -g。"""
    x = _as_node(x)
    out = Node(-x.data, requires_grad=x.requires_grad)
    _bind(out, [x], 'Neg')

    def _backward():
        if x.requires_grad:
            x.grad += -out.grad        # ∂(-x)/∂x = -1
    out._backward = _backward
    return out


def exp(x):
    """逐元素指数 y = eˣ。反向：dL/dx += y·g（eˣ 的导数是它自己）。"""
    x = _as_node(x)
    out = Node(np.exp(x.data), requires_grad=x.requires_grad)
    _bind(out, [x], 'Exp')

    def _backward():
        if x.requires_grad:
            x.grad += out.data * out.grad   # ∂eˣ/∂x = eˣ = y
    out._backward = _backward
    return out


def log(x):
    """逐元素自然对数 y = ln x。反向：dL/dx += g / x。"""
    x = _as_node(x)
    out = Node(np.log(x.data), requires_grad=x.requires_grad)
    _bind(out, [x], 'Log')

    def _backward():
        if x.requires_grad:
            x.grad += out.grad / x.data     # ∂ln(x)/∂x = 1/x
    out._backward = _backward
    return out


def pow(x, n):
    """逐元素幂 y = xⁿ（n 为标量；指数本身不可导，只对底数求导）。

    反向：dL/dx += n·xⁿ⁻¹·g。
    """
    x = _as_node(x)
    n_val = float(n.data) if isinstance(n, Node) else float(n)  # 标量指数
    out = Node(x.data ** n_val, requires_grad=x.requires_grad)
    _bind(out, [x], 'Pow')

    def _backward():
        if x.requires_grad:
            # 幂函数求导：d(xⁿ)/dx = n·xⁿ⁻¹
            x.grad += n_val * (x.data ** (n_val - 1.0)) * out.grad
    out._backward = _backward
    return out


def sqrt(x):
    """逐元素平方根 y = √x。反向：dL/dx += g/(2√x) = g/(2y)。"""
    x = _as_node(x)
    out = Node(np.sqrt(x.data), requires_grad=x.requires_grad)
    _bind(out, [x], 'Sqrt')

    def _backward():
        if x.requires_grad:
            x.grad += 0.5 * out.grad / out.data   # ∂√x/∂x = 1/(2√x)
    out._backward = _backward
    return out


# ============================================================================
#                                激活函数
# ============================================================================
def relu(x):
    """ReLU 激活 y = max(0, x)。反向：dL/dx += (x > 0)·g。

    说明：x=0 处次梯度约定为 0（工程惯例，不影响训练）。
    """
    x = _as_node(x)
    out = Node(np.maximum(x.data, 0.0), requires_grad=x.requires_grad)
    _bind(out, [x], 'Relu')

    def _backward():
        if x.requires_grad:
            # 正区导数 1、负区导数 0 -> 布尔掩码直接当 0/1 系数使用
            x.grad += (x.data > 0) * out.grad
    out._backward = _backward
    return out


def sigmoid(x):
    """Sigmoid 激活 y = 1/(1+e⁻ˣ)，数值稳定实现。反向：dL/dx += y(1-y)·g。

    数值稳定思路：
        令 ez = e^{-|z|} ∈ (0,1]（永不上溢），分两支：
        z ≥ 0: y = 1/(1+ez)；z < 0: y = ez/(1+ez)（等价于 eᶻ/(1+eᶻ)）。
    """
    x = _as_node(x)
    z = x.data
    ez = np.exp(-np.abs(z))                                   # 永不溢出的公共量
    out_data = np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))
    out = Node(out_data, requires_grad=x.requires_grad)
    _bind(out, [x], 'Sigmoid')

    def _backward():
        if x.requires_grad:
            # sigmoid 求导：y' = y(1-y)（用前向输出直接算，省一次指数）
            x.grad += out.data * (1.0 - out.data) * out.grad
    out._backward = _backward
    return out


def tanh(x):
    """双曲正切 y = tanh(x)。反向：dL/dx += (1 - y²)·g。"""
    x = _as_node(x)
    out = Node(np.tanh(x.data), requires_grad=x.requires_grad)
    _bind(out, [x], 'Tanh')

    def _backward():
        if x.requires_grad:
            # tanh 求导：y' = 1 - tanh²(x) = 1 - y²
            x.grad += (1.0 - out.data * out.data) * out.grad
    out._backward = _backward
    return out


def softmax(x, axis=-1):
    """Softmax y_i = e^{z_i - max} / Σ_j e^{z_j - max}（减最大值防溢出）。

    反向（雅可比向量积的经典化简）：
        ∂y_i/∂z_j = y_i(δ_ij - y_j)  =>  dL/dz = y ⊙ (g - Σ_j g_j y_j)
    """
    x = _as_node(x)
    z = x.data
    zmax = np.max(z, axis=axis, keepdims=True)     # 减最大值：数值稳定化
    e = np.exp(z - zmax)
    s = e.sum(axis=axis, keepdims=True)
    out_data = e / s
    out = Node(out_data, requires_grad=x.requires_grad)
    _bind(out, [x], 'Softmax')

    def _backward():
        if x.requires_grad:
            g = out.grad
            # dot = Σ_j g_j y_j（沿 softmax 轴）；这是 softmax 雅可比作用向量的化简式
            dot = (g * out.data).sum(axis=axis, keepdims=True)
            x.grad += out.data * (g - dot)
    out._backward = _backward
    return out


# ============================================================================
#                                卷积 / 池化
# ============================================================================
def gelu(x):
    """GELU 激活（tanh 近似，Transformer 标配）：
        y = 0.5·x·(1 + tanh(k·(x + 0.044715·x³)))，k = √(2/π)

    反向：令 t=tanh(u)，u=k(x+0.044715x³)，则
        dy/dx = 0.5(1+t) + 0.5·x·(1-t²)·k·(1+3·0.044715·x²)
    说明：精确 GELU 需要 erf，numpy 无内置导数；tanh 近似是 ViT/BERT 常用实现，
    与精确值最大差约 1e-3，对训练无实质影响。
    """
    x = _as_node(x)
    k = np.sqrt(2.0 / np.pi)
    u = k * (x.data + 0.044715 * x.data ** 3)     # 近似多项式的内层
    t = np.tanh(u)                                # 缓存 tanh，反向复用
    out = Node(0.5 * x.data * (1.0 + t), requires_grad=x.requires_grad)
    _bind(out, [x], 'GELU')

    def _backward():
        if x.requires_grad:
            du = k * (1.0 + 3.0 * 0.044715 * x.data ** 2)  # du/dx
            dy = 0.5 * (1.0 + t) + 0.5 * x.data * (1.0 - t ** 2) * du
            x.grad += dy * out.grad
    out._backward = _backward
    return out


def concat(nodes, axis=0):
    """沿指定轴拼接多个节点：y = concat([a,b,...], axis)。

    反向：按各输入在该轴上的长度切分上游梯度，分别累加回去。
    用途：ViT 把 [CLS] token 拼到 patch 序列前面。
    """
    nodes = [_as_node(n) for n in nodes]
    if not nodes:
        raise ValueError('concat 至少需要一个节点')
    ndim = nodes[0].data.ndim
    ax = axis if axis >= 0 else axis + ndim
    out = Node(np.concatenate([n.data for n in nodes], axis=ax),
               requires_grad=any(n.requires_grad for n in nodes))
    _bind(out, nodes, 'Concat')
    sizes = [n.data.shape[ax] for n in nodes]     # 每个输入在该轴上的长度

    def _backward():
        # np.split 的切分点 = 长度的累加前缀（去掉最后一个）
        pieces = np.split(out.grad, np.cumsum(sizes)[:-1], axis=ax)
        for n, g in zip(nodes, pieces):
            if n.requires_grad:
                n.grad += g
    out._backward = _backward
    return out


def broadcast_to(x, shape):
    """把节点广播到目标形状（只读扩展，不复制前向数据）。

    反向：沿被扩展的轴把梯度求和还原（复用 unbroadcast）。
    用途：把 (1,1,D) 的 [CLS] token 扩展到 (N,1,D)。
    """
    x = _as_node(x)
    out = Node(np.broadcast_to(x.data, shape).copy(),   # copy: 避免只读数组
               requires_grad=x.requires_grad)
    _bind(out, [x], 'BroadcastTo')

    def _backward():
        if x.requires_grad:
            x.grad += unbroadcast(out.grad, x.data.shape)
    out._backward = _backward
    return out


def select(x, index, axis):
    """沿 axis 取第 index 个切片（整数索引），如 select(x, 0, axis=1) 取 [CLS]。

    反向：新建与 x 同形的零梯度，把上游梯度散射到被选中的切片。
    """
    x = _as_node(x)
    ndim = x.data.ndim
    ax = axis if axis >= 0 else axis + ndim
    slicer = [slice(None)] * ndim
    slicer[ax] = index
    key = tuple(slicer)
    out = Node(np.array(x.data[key], copy=True), requires_grad=x.requires_grad)
    _bind(out, [x], 'Select')

    def _backward():
        if x.requires_grad:
            g = np.zeros_like(x.data)             # 未被选中的位置梯度为 0
            g[key] += out.grad
            x.grad += g
    out._backward = _backward
    return out


def conv2d(x, W, b=None, stride=1, padding=0):
    """二维卷积（im2col + 大矩阵乘实现，性能关键）。

    参数：
        x: Node/数组 (N, C, H, W) —— 输入 batch；
        W: Node/数组 (F, C, KH, KW) —— 卷积核；
        b: Node/数组 (F,) 或 (1,F) 或 None —— 输出通道偏置；
        stride (int): 步幅；
        padding (int): 零填充圈数。

    返回：
        Node: (N, F, OH, OW)，其中 OH=(H+2p-KH)//stride+1，OW 同理。

    前向实现（im2col）：
        col = im2col(x)                     -> (N*OH*OW, C*KH*KW)
        out = col @ W_col + b               -> (N*OH*OW, F)，W_col=(C*KH*KW, F)
        重排 -> (N, OH, OW, F) -> 轴置换 -> (N, F, OH, OW)

    反向推导（对 out2=(N*OH*OW,F) 记 g2）：
        dW：dL/dW_col = colᵀ @ g2  -> 还原 (F,C,KH,KW)
        db：dL/db = Σ_行 g2（每个输出通道对其所有输出位置求和）
        dx：dL/dcol = g2 @ W_colᵀ -> col2im 散射累加回 (N,C,H,W)
    """
    x, W = _as_node(x), _as_node(W)
    b = None if b is None else _as_node(b)
    xd = x.data                                  # (N,C,H,W)
    F, C, KH, KW = W.data.shape
    N, Cx, H, Wd = xd.shape
    if Cx != C:
        raise ValueError(f'conv2d 输入通道不匹配: x 通道 {Cx} vs W 通道 {C}')
    OH = (H + 2 * padding - KH) // stride + 1    # 输出高（向下取整）
    OW = (Wd + 2 * padding - KW) // stride + 1   # 输出宽
    if OH < 1 or OW < 1:
        raise ValueError(f'conv2d 输出尺寸非法: OH={OH}, OW={OW}（核对输入过大/步幅过大）')

    # ---------- 前向：im2col + 矩阵乘 ----------
    col = im2col(xd, KH, KW, stride, padding)    # (N*OH*OW, C*KH*KW)
    W_col = W.data.reshape(F, -1).T              # (C*KH*KW, F)：核摊平转置
    out2 = col @ W_col                           # (N*OH*OW, F)：核心矩阵乘
    if b is not None:
        out2 = out2 + b.data.reshape(-1)         # 偏置按最后一维(通道)广播
    # (N,OH,OW,F) -> (N,F,OH,OW)：把通道轴换到第 1 维
    out = Node(out2.reshape(N, OH, OW, F).transpose(0, 3, 1, 2),
               requires_grad=x.requires_grad or W.requires_grad
               or (b is not None and b.requires_grad))
    parents = [x, W] + ([] if b is None else [b])
    _bind(out, parents, 'Conv2D')

    def _backward():
        g = out.grad                                            # (N,F,OH,OW)
        # 把输出梯度拉直成矩阵 g2=(N*OH*OW, F)，与 forward 的 out2 对齐
        g2 = g.transpose(0, 2, 3, 1).reshape(N * OH * OW, F)
        if W.requires_grad:
            # dW_col = colᵀ @ g2 -> (C*KH*KW, F)，转置还原成核形状
            dW_col = col.T @ g2
            W.grad += dW_col.T.reshape(F, C, KH, KW)
        if b is not None and b.requires_grad:
            # db_f = Σ_{n,oh,ow} g2[.,f]：每个通道对所有输出位置求和
            b.grad += unbroadcast(g2.sum(axis=0), b.data.shape)
        if x.requires_grad:
            # dcol = g2 @ W_colᵀ -> (N*OH*OW, C*KH*KW)，再散射累加回图像
            dcol = g2 @ W_col.T
            x.grad += col2im(dcol, xd.shape, KH, KW, stride, padding)
    out._backward = _backward
    return out


def maxpool2d(x, kernel_size=2, stride=2):
    """二维最大池化。反向只把梯度放到前向最大值位置（记录 argmax）。

    参数：
        x: Node/数组 (N, C, H, W)；
        kernel_size (int|tuple): 窗口大小；
        stride (int): 步幅（无 padding）。

    返回：
        Node: (N, C, OH, OW)，OH=(H-KH)//stride+1。
    """
    x = _as_node(x)
    xd = x.data
    if isinstance(kernel_size, (tuple, list)):
        KH, KW = int(kernel_size[0]), int(kernel_size[1])
    else:
        KH = KW = int(kernel_size)
    stack, OH, OW = _pool_windows(xd, KH, KW, stride)   # (K,N,C,OH,OW)
    out_data = stack.max(axis=0)                        # 沿核偏移轴取最大
    argmax = stack.argmax(axis=0)                       # 记录每个窗口最大值位置（0..K-1）
    out = Node(out_data, requires_grad=x.requires_grad)
    _bind(out, [x], 'MaxPool2D')

    def _backward():
        g = out.grad                                    # (N,C,OH,OW)
        K = stack.shape[0]
        dstack = np.zeros_like(stack)
        for k in range(K):
            mask = (argmax == k)        # 布尔掩码：该核偏移是窗口最大值的位置
            dstack[k] = g * mask        # 梯度只流向最大值位置（其余位置梯度为 0）
        # 散射累加回输入（stride<kernel 时窗口重叠，+= 保证不丢）
        x.grad += _pool_scatter(dstack, xd.shape, KH, KW, stride)
    out._backward = _backward
    return out


def avgpool2d(x, kernel_size=2, stride=2):
    """二维平均池化。反向把梯度平均分配（/窗口面积）到窗口每个位置。"""
    x = _as_node(x)
    xd = x.data
    if isinstance(kernel_size, (tuple, list)):
        KH, KW = int(kernel_size[0]), int(kernel_size[1])
    else:
        KH = KW = int(kernel_size)
    stack, OH, OW = _pool_windows(xd, KH, KW, stride)   # (K,N,C,OH,OW)
    K = stack.shape[0]
    out = Node(stack.mean(axis=0), requires_grad=x.requires_grad)
    _bind(out, [x], 'AvgPool2D')

    def _backward():
        g = out.grad
        # 平均池化每个输出 = 窗口均值 -> 每个窗口位置的偏导都是 1/K
        dstack = np.zeros_like(stack)
        for k in range(K):
            dstack[k] = g / K               # 均匀分配到窗口每个位置
        x.grad += _pool_scatter(dstack, xd.shape, KH, KW, stride)
    out._backward = _backward
    return out


# ============================================================================
#                                形状 / 归约
# ============================================================================
def reshape(x, shape):
    """改变形状（元素总数不变）。反向：把梯度 reshape 回原形状。"""
    x = _as_node(x)
    out = Node(x.data.reshape(shape), requires_grad=x.requires_grad)
    _bind(out, [x], 'Reshape')

    def _backward():
        if x.requires_grad:
            x.grad += out.grad.reshape(x.data.shape)   # 梯度按原形状还原
    out._backward = _backward
    return out


def flatten(x, start_axis=1):
    """从 start_axis 起摊平：(N,C,H,W) -> (N, C*H*W)。反向等价 reshape 还原。"""
    x = _as_node(x)
    old_shape = x.data.shape
    # 目标形状 = start_axis 之前的维度保持 + 余下维度合并成 1 维
    head = old_shape[:start_axis]
    tail = int(np.prod(old_shape[start_axis:])) if old_shape[start_axis:] else 1
    return reshape(x, head + (tail,))


def transpose(x, axes=None):
    """轴重排 y = x.transpose(axes)。反向：梯度做逆置换。

    推导：若 y = perm(x)，则 x = perm⁻¹(y)，故 dL/dx = g.perm⁻¹（argsort(axes)）。
    """
    x = _as_node(x)
    out = Node(x.data.transpose(axes), requires_grad=x.requires_grad)
    _bind(out, [x], 'Transpose')

    def _backward():
        if x.requires_grad:
            if axes is None:
                # 完全逆序的 transpose 是自逆的：再 transpose 一次即还原
                x.grad += out.grad.transpose()
            else:
                # 逆置换 = argsort(axes)
                x.grad += out.grad.transpose(np.argsort(axes))
    out._backward = _backward
    return out


def _expand_reduced_grad(g, x_shape, axis, keepdims):
    """把归约结果的梯度 g 扩展回输入形状（供 sum/mean 反向广播）。

    实现：keepdims=False 时先在归约轴上补回 size=1 的维度，再整体广播。
    """
    if axis is None:
        # 全量归约 -> g 是标量（0 维），直接广播到整个输入形状
        return np.broadcast_to(g, x_shape)
    axes = axis if isinstance(axis, (tuple, list)) else (axis,)
    # 负轴规范化为正轴，并升序排列（逐个 expand_dims 时索引才正确）
    axes = tuple(sorted(a % len(x_shape) for a in axes))
    if not keepdims:
        for a in axes:
            g = np.expand_dims(g, a)      # 把被归约掉的轴补回来（size=1）
    return np.broadcast_to(g, x_shape)


def sum(x, axis=None, keepdims=False):
    """沿轴求和 y = Σ x。反向：dL/dx += 广播(g)（求和的偏导恒为 1）。

    注意：函数名与内置 sum 冲突，请通过 `import miniflows.ops` 方式使用。
    """
    x = _as_node(x)
    out = Node(x.data.sum(axis=axis, keepdims=keepdims),
               requires_grad=x.requires_grad)
    _bind(out, [x], 'Sum')

    def _backward():
        if x.requires_grad:
            # 求和把多份元素合并成一份 -> 反向把梯度广播回每一份
            x.grad += _expand_reduced_grad(out.grad, x.data.shape, axis, keepdims)
    out._backward = _backward
    return out


def mean(x, axis=None, keepdims=False):
    """沿轴求均值 y = Σx/n。反向：dL/dx += 广播(g)/n（n 为归约元素个数）。"""
    x = _as_node(x)
    out = Node(x.data.mean(axis=axis, keepdims=keepdims),
               requires_grad=x.requires_grad)
    _bind(out, [x], 'Mean')

    def _backward():
        if x.requires_grad:
            # 均值 = 求和 / n -> 偏导 1/n；n = 输入元素数 / 输出元素数
            # 注意：本模块定义了同名算子 max/sum，会遮蔽 Python 内置函数，
            #       所以这里用条件表达式代替 max(out.data.size, 1)。
            n = x.data.size // (out.data.size if out.data.size > 0 else 1)
            x.grad += _expand_reduced_grad(out.grad, x.data.shape,
                                           axis, keepdims) / n
    out._backward = _backward
    return out


def max(x):
    """全量最大值归约（返回标量 Node）。梯度只流向最大值位置。

    说明：若存在并列最大值，梯度流向所有并列位置（次梯度约定），
    数值检验时请使用无重复输入。
    """
    x = _as_node(x)
    out = Node(x.data.max(), requires_grad=x.requires_grad)
    _bind(out, [x], 'Max')

    def _backward():
        if x.requires_grad:
            # 掩码：等于最大值的位置为 True -> 梯度放到最大值位置
            x.grad += (x.data == out.data) * out.grad
    out._backward = _backward
    return out
