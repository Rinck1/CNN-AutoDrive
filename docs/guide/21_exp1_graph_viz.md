# `experiments/exp1_graph_viz.py` 阅读指南

> 前置：`graph.py`、`visualization.py` ｜ 难度：★ ｜ 建议 20 分钟 ｜ 扩展①

## 1. 文件定位

计算图可视化演示，两幕：

1. 手算表达式 `z=(a+b)*(b+c)` 的图 + 梯度验证；
2. MLP(4→16→3)+CrossEntropyLoss 的完整前向图。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `demo_expression_graph` (50) | 表达式图 + dz/da 等 |
| 2 | `demo_mlp_graph` (88) | MLP 前向图 |
| 3 | `__main__` | 调用两幕并打印节点/边数 |

## 3. 逐段讲解

### 3.1 表达式图

```python
a,b,c = param(2.), param(3.), param(1.)
s1,s2 = a+b, b+c
z = s1*s2
z.backward()
```

- 解析：`dz/da=4, dz/db=9, dz/dc=5`；
- 打印框架梯度，与手算对照；
- 节点 6、边 6。

### 3.2 MLP 图

```python
X = Node(randn(8,4)); y = ...
model = Sequential(Linear(4,16), ReLU(), Linear(16,3))
loss = CrossEntropyLoss()(model(X), y)
draw_computation_graph(loss, '计算图_MLP.png')
```

- 节点 11、边 10；
- 图中能看到参数节点、MatMul/BroadcastAdd/ReLU/CrossEntropyLoss。

### 3.3 产物

- `outputs/exp1/计算图_表达式.png`（约 994×706）；
- `outputs/exp1/计算图_MLP.png`（约 1837×967）。

## 4. 调用关系

`graph.param → ops 建图 → visualization.draw_computation_graph`；
本实验是 exp1 的唯一入口。

## 5. 易错点

1. 图片尺寸异常：绘图函数必须显式设置 xlim/ylim（已在 visualization 修复）；
2. 节点太多：`max_nodes` 截断；
3. 参数名：`param(..., name='a')` 才会在图上显示名字；
4. 标量节点形状显示为“标量”。

## 6. 自检问题

1. 手推三个梯度并解释每一项来自哪条路径；
2. 图中箭头方向是父→子还是子→父？
3. 如果去掉 `requires_grad`，图会少哪些节点？
