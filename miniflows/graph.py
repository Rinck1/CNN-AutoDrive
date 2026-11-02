# -*- coding: utf-8 -*-
"""
miniflows/graph.py —— MiniFlows 计算图核心：Node 节点与图工具函数
==================================================================

【模块定位】
    本模块实现 MiniFlows 框架的基石 —— 计算图节点 ``Node``。整个框架采用
    "动态计算图 + 闭包反向传播" 的设计（PyTorch / micrograd 风格）：

    * 前向：每次运算生成一个新的 Node，通过 ``parents`` 指针记录其输入，
      从而把整次前向计算隐式地组织成一棵 DAG（有向无环图）；
      无全局 Graph 对象，图的存亡跟随 Python 引用，用完即弃。
    * 反向：每个非叶子节点持有 ``_backward`` 闭包，闭包内写死了
      "输出梯度 -> 各输入梯度" 的局部求导规则；从损失节点调用
      ``backward()`` 时先做拓扑排序，再逆序执行各节点的 ``_backward``，
      即可按链式法则把损失对全部参数的梯度一步步"拉"回来。

【数学原理：反向传播 = 链式法则的逆序应用】
    设前向计算中有 y = f(x1, x2, ..., xk)，且已知损失对 y 的梯度 g = dL/dy，
    则链式法则给出
        dL/dxi = g · (∂f/∂xi)        （雅可比矩阵 × 上游梯度向量）
    每个算子的 ``_backward`` 闭包只负责实现上式中的"局部雅可比 × g"；
    框架负责把所有局部规则按拓扑序（逆序）串起来，即完成整体自动微分。

【设计约定】
    1. Node 内部数据统一 ``np.float64`` —— 保证数值梯度检验（中心差分）精度；
    2. 梯度累加使用 ``+=`` —— 同一节点被多条路径使用时（如 y = x*x + x），
       各路径贡献的梯度相加，与链式法则的"多路径求和"一致，不会丢失；
    3. 运算符重载（+ - * / ** @ 等）遇到 python 数字 / np.ndarray 时
       自动包装为常量节点（requires_grad=False），实现与标量/数组无缝混算；
    4. ``requires_grad=False`` 的节点不参与反向（在拓扑遍历时整枝剪掉），
       不影响正确性，还能节省计算。

【依赖】仅 numpy。本文件由核心模块组维护。
"""

import numpy as np


# ============================================================================
#                                计算图节点
# ============================================================================
class Node:
    """计算图节点：同时承载"前向数据"与"反向梯度"，是框架的一等公民。

    属性说明
    --------
    data : np.ndarray
        前向计算值，统一为 np.float64 数组（标量也是 0 维数组）。
    grad : np.ndarray | None
        反向传播得到的梯度，形状与 data 相同；backward() 前为 None。
    requires_grad : bool
        True 表示该节点是可训练参数（或依赖参数的计算结果），需要求梯度。
    parents : list[Node]
        父节点（即本节点的输入），构成计算图 DAG 的边。
    op_name : str
        产生本节点的算子名（Pascal 命名，如 'MatMul'）；
        叶子参数为 'Parameter'，常量为 'Constant'。
    name : str | None
        参数名 / 调试名（参数保存、可视化、summary 都会用到）。
    """

    def __init__(self, data, requires_grad=False, name=None):
        """构造节点。

        参数：
            data: np.ndarray 或 python 标量/列表 -> 统一转 np.float64 数组；
            requires_grad (bool): True 表示可训练参数；
            name (str|None): 参数名/调试名。
        """
        # 统一数据类型为 float64：数值梯度检验需要 ~1e-10 级精度，float32 不够
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None                       # 反向传播前没有梯度
        self.requires_grad = bool(requires_grad)
        self.parents = []                      # 父节点列表（由算子函数填写）
        # 叶子节点类型标记：参数 / 常量；算子输出会在算子函数里覆盖为算子名
        self.op_name = 'Parameter' if requires_grad else 'Constant'
        self.name = name
        self._backward = None                  # 反向闭包（由算子函数注册）

    # ------------------------------------------------------------------ 基础
    @property
    def shape(self):
        """返回前向数据的形状 tuple（仿 numpy 接口，方便使用）。"""
        return self.data.shape

    def zero_grad(self):
        """把本节点的梯度清空（置 None）。"""
        self.grad = None

    def detach(self):
        """返回一个脱离计算图的新 Node（数据复制、无父节点、不求梯度）。

        用途：把中间结果当作纯数据使用（如可视化、保存），切断梯度流动。
        """
        new = Node(self.data.copy(), requires_grad=False, name=self.name)
        new.op_name = self.op_name    # 保留算子名信息便于调试
        return new

    # -------------------------------------------------------------- 反向传播
    def backward(self, grad=None):
        """从本节点开始反向传播（链式法则 + 拓扑排序逆序执行）。

        参数：
            grad: 输出节点的种子梯度 dL/d(本节点)。None 时取全 1
                  （等价于假设 L = 本节点，因此 dL/dy = 1）。

        实现思路：
            1) 从本节点出发，只遍历 requires_grad=True 的子图，做拓扑排序
               （输入在前、本节点最后）；不需要梯度的子树整枝剪掉；
            2) 给拓扑序中所有节点初始化零梯度（已有梯度则保留 —— 因此
               多次调用 backward 会对同一图累加梯度，语义同 PyTorch）；
            3) 本节点的 grad 覆盖为种子梯度；
            4) 按拓扑序逆序逐节点执行 _backward 闭包 —— 每个闭包把
               "本节点梯度 × 局部雅可比" 用 += 累加进各父节点。
        """
        if not self.requires_grad:
            # 常量节点没有上游梯度可言，直接返回（契约：不参与反向不影响正确性）
            return

        # ---- 1. 子图拓扑排序（迭代实现，避免深图递归爆栈）----
        topo = []                 # 拓扑序：父节点（输入）在前，self 在最后
        visited = set()           # 已入栈过的节点集合（按 id 判重）
        stack = [(self, False)]   # (节点, 是否已展开过子节点)
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)          # 所有父节点都已入 topo，此时才轮到自己
                continue
            # 不需要梯度的节点直接剪枝（其整棵父图都不需要梯度）
            if id(node) in visited or not node.requires_grad:
                continue
            visited.add(id(node))
            stack.append((node, True))     # 后处理：等父节点全部完成后追加
            for p in node.parents:
                stack.append((p, False))   # 先展开父节点
        # ---- 2. 初始化梯度（已存在的保留 -> 多次 backward 自动累加）----
        for v in topo:
            if v.grad is None:
                v.grad = np.zeros_like(v.data)
        # ---- 3. 写入种子梯度 ----
        if grad is None:
            grad = np.ones_like(self.data)              # 种子默认全 1
        else:
            grad = np.asarray(grad, dtype=np.float64)
            if grad.shape != self.data.shape:
                grad = np.broadcast_to(grad, self.data.shape)  # 允许标量种子广播
        self.grad = grad
        # ---- 4. 拓扑序逆序执行各节点的反向闭包 ----
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()

    # ------------------------------------------------------ 运算符重载（混算）
    # 以下运算符遇到 python 数字 / np.ndarray 时自动包装为常量 Node，
    # 全部转发给 miniflows.ops 中的算子函数（惰性导入避免循环依赖）。
    def __add__(self, other):
        """self + other：逐元素加（other 可为 Node/标量/ndarray）。"""
        return _ops().add(self, _wrap(other))

    def __radd__(self, other):
        """other + self（other 为标量/ndarray 时触发）。"""
        return _ops().add(_wrap(other), self)

    def __sub__(self, other):
        """self - other：逐元素减。"""
        return _ops().sub(self, _wrap(other))

    def __rsub__(self, other):
        """other - self（other 为标量/ndarray 时触发）。"""
        return _ops().sub(_wrap(other), self)

    def __mul__(self, other):
        """self * other：逐元素乘。"""
        return _ops().mul(self, _wrap(other))

    def __rmul__(self, other):
        """other * self（other 为标量/ndarray 时触发）。"""
        return _ops().mul(_wrap(other), self)

    def __truediv__(self, other):
        """self / other：逐元素除。"""
        return _ops().div(self, _wrap(other))

    def __rtruediv__(self, other):
        """other / self（other 为标量/ndarray 时触发）。"""
        return _ops().div(_wrap(other), self)

    def __neg__(self):
        """-self：逐元素取负。"""
        return _ops().neg(self)

    def __pow__(self, n):
        """self ** n：逐元素幂（n 为标量）。"""
        return _ops().pow(self, n)

    def __matmul__(self, other):
        """self @ other：矩阵乘。"""
        return _ops().matmul(self, _wrap(other))

    # ------------------------------------------------------- 常用方法（转发 ops）
    def sum(self, axis=None, keepdims=False):
        """沿指定轴求和（归约）。axis=None 表示全部求和成标量。"""
        return _ops().sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        """沿指定轴求均值（归约）。"""
        return _ops().mean(self, axis, keepdims)

    def max(self):
        """全量最大值归约（返回标量 Node）；梯度只流向最大值位置。"""
        return _ops().max(self)

    def exp(self):
        """逐元素指数 e^x。"""
        return _ops().exp(self)

    def log(self):
        """逐元素自然对数 ln(x)。"""
        return _ops().log(self)

    def sqrt(self):
        """逐元素平方根 √x。"""
        return _ops().sqrt(self)

    def reshape(self, shape):
        """改变形状（总元素数不变），梯度反向时自动还原形状。"""
        return _ops().reshape(self, shape)

    def flatten(self, start_axis=1):
        """从 start_axis 起摊平成 1 维，如 (N,C,H,W)->(N,C*H*W)。"""
        return _ops().flatten(self, start_axis)

    def transpose(self, axes=None):
        """轴重排（axes=None 表示完全逆序），梯度反向时做逆置换。"""
        return _ops().transpose(self, axes)

    def __repr__(self):
        """调试用字符串：算子名/名字/形状/是否求梯度。"""
        return (f"Node(op={self.op_name}, name={self.name}, "
                f"shape={self.shape}, requires_grad={self.requires_grad})")


# ============================================================================
#                                图工具函数
# ============================================================================
def _wrap(x):
    """把标量 / ndarray / list 包装为常量 Node（Node 原样返回）。

    参数：
        x: Node | 数字 | np.ndarray | list

    返回：
        Node（常量，requires_grad=False）
    """
    if isinstance(x, Node):
        return x
    return Node(x)   # 默认 requires_grad=False -> 常量节点


def constant(array, name=None):
    """创建常量节点（requires_grad=False, op_name='Constant'）。

    参数：
        array: np.ndarray / 标量 / list；
        name: 调试名。
    返回：
        Node 常量节点。
    """
    return Node(array, requires_grad=False, name=name)


def param(array, name=None):
    """创建可训练参数节点（requires_grad=True, op_name='Parameter'）。

    参数：
        array: np.ndarray（初始参数值）；
        name: 参数名（保存/加载、可视化、named_params 都依赖它）。
    返回：
        Node 参数节点。
    """
    return Node(array, requires_grad=True, name=name)


def build_topo(root):
    """从 root 出发做完整图的拓扑排序（不看 requires_grad，纯结构遍历）。

    参数：
        root: Node，图的输出节点。
    返回：
        list[Node]：拓扑序，输入（叶子）在前，root 在最后。

    实现思路：
        迭代式后序 DFS（显式栈 + visited 判重），避免深图递归爆栈；
        节点只有在"全部父节点都已进入拓扑序"后才被追加。
    """
    topo = []
    visited = set()
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            topo.append(node)           # 父节点都已处理完，追加自己
            continue
        if id(node) in visited:
            continue                    # 同一节点多次入栈只处理一次
        visited.add(id(node))
        stack.append((node, True))      # 压回栈：等父节点全部展开后再追加
        for p in node.parents:
            stack.append((p, False))    # 先展开父节点
    return topo


def collect_edges(root):
    """收集 root 可达子图的所有边，供可视化模块画 DAG。

    参数：
        root: Node，图的输出节点。
    返回：
        list[tuple]：[(parent, child, op_name), ...]，
        op_name 为产生 child 的算子名。
    """
    edges = []
    visited = {id(root)}      # 从 root 开始 BFS
    queue = [root]
    while queue:
        child = queue.pop(0)
        for p in child.parents:
            # 每条边记录 (输入节点, 输出节点, 输出节点的算子名)
            edges.append((p, child, child.op_name))
            if id(p) not in visited:
                visited.add(id(p))
                queue.append(p)
    return edges


# ============================================================================
#                     惰性加载 ops 模块（避免循环导入）
# ============================================================================
_OPS = None


def _ops():
    """惰性获取 miniflows.ops 模块对象。

    说明：ops.py 需要 import 本模块的 Node，而本模块的运算符重载/方法
    又要调用 ops 的算子函数 —— 若在文件头互相 import 会形成循环导入。
    因此这里在"首次真正调用算子"时才加载 ops（此时 ops 必已加载完毕）。
    """
    global _OPS
    if _OPS is None:
        from miniflows import ops as _ops_module
        _OPS = _ops_module
    return _OPS
