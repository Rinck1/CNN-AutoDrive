# -*- coding: utf-8 -*-
"""
experiments/exp1_graph_viz.py —— 扩展①：计算图可视化演示
============================================================

演示 MiniFlows 的计算图可视化功能（纯 matplotlib 实现，不依赖 graphviz）：

实验 1：手算表达式 z = (a+b)*(b+c) 的计算图
    - 用 param() 创建参数 a/b/c，通过运算符重载自动建图；
    - 调用 z.backward() 反向传播，验证梯度解析解：
        z = (a+b)(b+c)  =>  dz/da = b+c, dz/db = (b+c)+(a+b), dz/dc = a+b
    - 保存 计算图_表达式.png。

实验 2：一个完整 MLP(4 -> 16 -> 3) + CrossEntropyLoss 前向的计算图
    - Sequential 容器搭建两层 MLP，随机数据前向并计算损失；
    - 展示真实网络的 DAG：参数节点（橙）/ 输入常量（灰）/ 运算（蓝）/ 损失（红）；
    - 保存 计算图_MLP.png。

运行方式（项目根目录）：python experiments/exp1_graph_viz.py
输出目录：outputs/exp1/
"""

import os
import sys

# ------------------ 标准脚本初始化头（契约第 1 节） ------------------
try:  # Windows 控制台强制 UTF-8，避免中文乱码报错
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # 使脚本能 import miniflows / common_setup
from common_setup import init_env
init_env(seed=42)  # 统一设置：随机种子 + matplotlib 中文字体 + Agg 后端
# ---------------------------------------------------------------------

import numpy as np

from miniflows.graph import param, build_topo, collect_edges  # 节点构造与图遍历
from miniflows.layers import Sequential, Linear, ReLU         # MLP 结构
from miniflows.losses import CrossEntropyLoss                 # 多分类损失
from miniflows.visualization import draw_computation_graph    # 扩展① 核心 API

OUT_DIR = os.path.join(_ROOT, "outputs", "exp1")              # 契约第 12 节约定的输出目录
os.makedirs(OUT_DIR, exist_ok=True)                           # 自动创建输出目录


def demo_expression_graph():
    """实验 1：手算表达式 z=(a+b)*(b+c) 的计算图 + 梯度解析解验证。"""
    print("\n" + "=" * 60)
    print("实验 1：表达式 z=(a+b)*(b+c) 的计算图")
    print("=" * 60)

    # ---------- 1. 用参数节点构造表达式（运算符重载自动建图） ----------
    a = param(2.0, name="a")                                # 参数 a（requires_grad=True）
    b = param(3.0, name="b")                                # 参数 b
    c = param(1.0, name="c")                                # 参数 c
    s1 = a + b                                              # s1 = a+b（Add 节点）
    s2 = b + c                                              # s2 = b+c（Add 节点）
    z = s1 * s2                                             # z = s1*s2（Mul 节点，根节点）

    # ---------- 2. 反向传播并对比解析解 ----------
    z.backward()                                            # 从根节点 z 反向求全图梯度
    # 解析解：z=(a+b)(b+c) => dz/da=b+c=4, dz/db=(b+c)+(a+b)=4+5=9, dz/dc=a+b=5
    print(f"前向: s1=a+b={float(s1.data):.1f}, s2=b+c={float(s2.data):.1f}, "
          f"z={float(z.data):.1f}")
    print(f"dz/da = {float(a.grad):.4f}（解析解 b+c = 4）")
    print(f"dz/db = {float(b.grad):.4f}（解析解 (b+c)+(a+b) = 9）")
    print(f"dz/dc = {float(c.grad):.4f}（解析解 a+b = 5）")
    # 自动微分与手算结果应完全一致（float64 精度）
    assert abs(float(a.grad) - 4.0) < 1e-12, "dz/da 与解析解不符"
    assert abs(float(b.grad) - 9.0) < 1e-12, "dz/db 与解析解不符"
    assert abs(float(c.grad) - 5.0) < 1e-12, "dz/dc 与解析解不符"
    print("[通过] 自动微分梯度与手算解析解完全一致")

    # ---------- 3. 统计图规模并保存可视化 ----------
    nodes = build_topo(z)                                   # 拓扑排序取全部节点
    edges = collect_edges(z)                                # 收集全部有向边
    print(f"图规模：节点 {len(nodes)} 个，边 {len(edges)} 条")
    path = os.path.join(OUT_DIR, "计算图_表达式.png")         # 输出文件（中文文件名）
    draw_computation_graph(z, path, title="计算图：z = (a+b) * (b+c)")
    print(f"[已保存] {path}")
    return len(nodes), len(edges)


def demo_mlp_graph():
    """实验 2：MLP(4->16->3) + CrossEntropyLoss 前向计算图可视化。"""
    print("\n" + "=" * 60)
    print("实验 2：MLP(4->16->3) + CrossEntropyLoss 的前向计算图")
    print("=" * 60)

    # ---------- 1. 搭建两层 MLP 并做一次前向 ----------
    model = Sequential(
        Linear(4, 16, name="fc1"),                          # 第一层：4 -> 16
        ReLU(),                                             # 非线性激活
        Linear(16, 3, name="fc2"),                          # 第二层：16 -> 3（3 分类 logits）
    )
    loss_fn = CrossEntropyLoss()                            # 多分类交叉熵损失
    rs = np.random.RandomState(42)                          # 独立随机源生成演示数据
    X = rs.randn(8, 4)                                      # 批量 8 的 4 维输入
    y = rs.randint(0, 3, size=8)                            # 8 个整数标签（0/1/2）

    logits = model(X)                                       # 前向：Sequential 自动逐层建图
    loss = loss_fn(logits, y)                               # 损失节点（本实验的图根节点）
    print(f"前向: logits 形状 {logits.shape}, loss = {float(loss.data):.4f}")

    # ---------- 2. 统计图规模并保存可视化 ----------
    nodes = build_topo(loss)                                # 以损失为根遍历全图
    edges = collect_edges(loss)                             # 全部有向边
    print(f"图规模：节点 {len(nodes)} 个，边 {len(edges)} 条")
    path = os.path.join(OUT_DIR, "计算图_MLP.png")
    draw_computation_graph(loss, path, title="计算图：MLP(4-16-3) + CrossEntropyLoss")
    print(f"[已保存] {path}")
    return len(nodes), len(edges)


if __name__ == "__main__":
    n1, e1 = demo_expression_graph()                        # 实验 1：表达式图
    n2, e2 = demo_mlp_graph()                               # 实验 2：MLP 图
    print("\n" + "=" * 60)
    print("扩展① 计算图可视化演示完成！")
    print(f"  表达式图: {n1} 节点 / {e1} 边 -> outputs/exp1/计算图_表达式.png")
    print(f"  MLP 图:   {n2} 节点 / {e2} 边 -> outputs/exp1/计算图_MLP.png")
    print("=" * 60)
