# `miniflows/graph.py` 阅读指南

> 前置：numpy 基础、链式法则 ｜ 难度：★★★ ｜ 建议 90 分钟 ｜ 必读

## 1. 文件定位

这是 MiniFlows 的**心脏**：定义计算图节点 `Node`、前向建图、拓扑排序、反向传播。
不理解这个文件，后面的 ops/layers/losses 都只能死记。

核心思想：**define-by-run**。没有全局 Graph 对象，图由 `Node.parents` 引用隐式构成；
每个非叶子节点持有一个 `_backward` 闭包，反向时从损失节点逆序执行。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 | 必读 |
|---|---|---|---|
| 1 | `Node.__init__` (63) | 保存 data/grad/parents/op_name/name | ★★★ |
| 2 | `Node.backward` (101) | 拓扑排序 + 逆序执行 `_backward` | ★★★ |
| 3 | 运算符重载 (159+) | `+ - * / @ **` 与反射版本 | ★★★ |
| 4 | `_wrap` (253) | 把标量/ndarray 包装成常量 Node | ★★ |
| 5 | `constant` (267) / `param` (279) | 常量节点 / 可训练参数节点 | ★★ |
| 6 | `build_topo` (291) | 返回拓扑序（输入在前） | ★★ |
| 7 | `collect_edges` (320) | 返回 `(parent, child, op_name)` 供可视化 | ★ |

## 3. 逐段讲解

### 3.1 Node 的四个关键字段

```python
self.data = np.asarray(data, dtype=np.float64)  # 前向值
self.grad = None                                # 反向梯度
self.parents = []                               # 输入节点（DAG 边）
self._backward = None                           # 局部反向闭包
```

- 统一 float64：数值梯度检验需要 ~1e-10 精度，float32 不够；
- `requires_grad=True` 的参数节点在反向时才会被遍历；
- `op_name` 用于可视化/调试（如 `MatMul`、`Conv2D`）。

### 3.2 backward 的五步

1. 从 `self` 做迭代后序 DFS（避免深图递归爆栈）；
2. 只遍历 `requires_grad=True` 的子图，常量分支整枝剪掉；
3. 给拓扑序所有节点初始化零梯度（已有 grad 则保留 → 多次 backward 累加）；
4. 写入种子梯度（缺省全 1，可传外部 grad）；
5. 逆序遍历，执行每个节点的 `_backward()`。

伪代码：

```python
topo = []
stack = [(self, False)]
while stack:
    node, processed = stack.pop()
    if processed: topo.append(node); continue
    if id(node) in visited or not node.requires_grad: continue
    visited.add(id(node)); stack.append((node, True))
    for p in node.parents: stack.append((p, False))
for v in topo:
    if v.grad is None: v.grad = np.zeros_like(v.data)
self.grad = np.ones_like(self.data) if grad is None else grad
for v in reversed(topo):
    if v._backward: v._backward()
```

**为什么梯度用 `+=`**：一个节点可能被多个子节点使用，例如 `y = x*x + x`，
两条路径的梯度都要累加，否则只会得到其中一份。

### 3.3 运算符重载与 `__array_priority__`

`x + 1.5`、`2.5 * x`、`x @ W` 都转发给 `miniflows.ops` 的对应函数，
标量/ndarray 先经 `_wrap` 包成常量 Node。

关键一行：

```python
__array_priority__ = 1000
```

没有它，`np.ones((2,2)) - node` 会被 numpy 当成 object 数组逐元素处理，
得到“装满 Node 的 ndarray”而不是一个 Node。设置高优先级后 numpy 会让出运算权，
调用 `Node.__rsub__`。

### 3.4 constant / param

- `constant(array)`：`requires_grad=False`，只参与前向；
- `param(array, name=...)`：`requires_grad=True`，可被优化器收集；
- 叶子节点的 `op_name` 分别为 `Constant` / `Parameter`。

### 3.5 build_topo / collect_edges

- `build_topo(root)`：只含 requires_grad 子图的拓扑序，最后一个一定是 root；
- `collect_edges(root)`：遍历 parents，产出 `(p, n, n.op_name)`，
  供 `visualization.draw_computation_graph` 画箭头。

## 4. 调用关系与数据流

```
ops.add(x, y) → Node(data=x+y, parents=[x,y]) + _backward
                       │
loss.backward() ───────┘
   ├─ build Topo (graph.py)
   ├─ init grads
   └─ reverse loop → 每个 _backward 把梯度 += 到 parents
```

数据形状：Node 本身不限制形状，形状语义由算子负责；所有中间节点都是 ndarray。

## 5. 易错点/设计取舍

1. **梯度累加语义**：训练每一步前必须 `optimizer.zero_grad()`；
2. **常量不积累梯度**：`c.grad` 永远是 None；
3. **`backward()` 不自动清图**：计算图是 Python 对象引用，loss 释放后自然回收；
4. **非 requires_grad 的 root 直接返回**：不会报错；
5. **float64 是刻意选择**：改成 float32 会让梯度检验误差从 1e-9 升到 1e-3 量级。

## 6. 自检问题

1. 画 `z=(a+b)*(b+c)` 的 DAG，并手推 `dz/da=4, dz/db=9, dz/dc=5`；
2. 为什么 `x*x+x` 的梯度是 `2x+1`？如果 `+=` 写成 `=` 会得到什么？
3. `__array_priority__` 解决什么具体 bug？构造一个复现例子；
4. 拓扑序为什么必须“父节点在前、root 最后”？
5. 连续调用两次 `loss.backward()` 不清零，梯度会怎样？与 PyTorch 一致吗？
