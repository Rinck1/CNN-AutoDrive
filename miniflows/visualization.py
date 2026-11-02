# -*- coding: utf-8 -*-
"""
miniflows/visualization.py —— 可视化模块（扩展①：计算图可视化 + 通用画图）
================================================================================

本模块提供两大类能力：

1. 计算图可视化（扩展①核心）：
   draw_computation_graph —— 纯 matplotlib 实现的 DAG（有向无环图）绘制：
   - 按拓扑深度分层布局（x 轴 = 深度，输入在左、输出/损失在右）；
   - 节点框内显示 参数名 / 算子名 + 张量形状；
   - 有向箭头连接 parent -> child；
   - 参数节点（橙）、常量节点（灰）、普通运算（蓝）、损失节点（红）分色显示；
   - 不依赖 graphviz，Windows 环境零额外安装。

2. 通用训练画图：
   plot_loss_curve —— 训练/验证损失（左轴）+ 准确率（右轴）双轴曲线；
   plot_lines —— 通用多曲线对比图；
   plot_images_grid —— 图像网格展示（MNIST 等）。

设计说明：draw_computation_graph 采用"鸭子类型"遍历——只要节点对象拥有
``parents`` / ``op_name`` / ``name`` / ``data``（或 ``shape``）属性即可绘制，
因此不 import miniflows.graph，可以在核心框架未就绪时独立测试，
也能绘制任意符合 Node 契约的对象。
"""

import math
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # 图例用的色块代理


# ======================================================================
# 一、中文字体配置
# ======================================================================
def setup_chinese_font():
    """配置 matplotlib 中文显示：指定中文字体候选列表 + 负号正常显示。

    说明：common_setup.init_env 已调用本函数，脚本通常无需重复调用；
    保留为公开 API 便于单独使用 visualization 时手动开启中文。
    """
    # Windows 自带中文字体按优先级尝试，第一个存在的生效
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                                       "KaiTi", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False  # 用 ASCII 负号，避免中文字体缺 U+2212 报警


def _ensure_chinese_font():
    """画图前的字体兜底检查：若仍是默认西文字体（DejaVu Sans）则自动开启中文。

    这样即使调用方忘记 init_env / setup_chinese_font（例如单元测试），
    图中的中文标题也不会变成"口口"方块。
    """
    current = plt.rcParams.get("font.sans-serif") or []
    if not current or current[0] == "DejaVu Sans":          # 未配置过中文字体
        setup_chinese_font()


# ======================================================================
# 二、计算图遍历工具（不依赖 miniflows.graph 的本地实现）
# ======================================================================
def _local_topo(root):
    """从 root 沿 parents 反向遍历，得到拓扑序列表（输入在前，root 在最后）。

    实现思路：后序 DFS（迭代栈实现，避免深图递归爆栈）——
    先把一个节点的所有父节点压栈处理完，再输出该节点，天然满足拓扑序。

    参数：
        root (Node): 计算图输出节点（如损失节点）。

    返回：
        list[Node]: 拓扑排序后的节点列表。
    """
    order, visited, stack = [], set(), [root]
    while stack:                                            # 迭代式后序 DFS
        node = stack[-1]                                    # 只查看不弹出
        if id(node) in visited:                             # 已完成：弹出并记录
            stack.pop()
            order.append(node)
            continue
        visited.add(id(node))                               # 标记"进行中"
        for p in getattr(node, "parents", []) or []:        # 先压入所有未访问的父节点
            if id(p) not in visited:
                stack.append(p)
    return order                                            # 父节点总是先于子节点出现


def _collect_edges_local(nodes):
    """从节点集合收集所有边 (parent, child)（去重）。

    参数：
        nodes (list[Node]): 拓扑序节点列表。

    返回：
        list[tuple[Node, Node]]: 边列表，parent 指向 child。
    """
    ids = {id(n) for n in nodes}                            # 参与绘制的节点 id 集合
    edges, seen = [], set()
    for n in nodes:                                         # 每个节点的每条入边
        for p in getattr(n, "parents", []) or []:
            if id(p) in ids:                                # 只保留两端都被绘制的边
                key = (id(p), id(n))
                if key not in seen:                         # 同一对节点可能有重复父引用
                    seen.add(key)
                    edges.append((p, n))
    return edges


def _node_shape_str(node):
    """取节点形状描述字符串（优先 .shape 属性，退化到 data.shape）。"""
    shape = getattr(node, "shape", None)                    # 契约规定 Node 有 shape 属性
    if shape is None:                                       # 鸭子类型退化：直接看 data
        data = getattr(node, "data", None)
        shape = getattr(data, "shape", ())
    if shape == () or shape is None:                        # 标量节点用中文标注更直观
        return "标量" if shape == () else "-"
    return "x".join(str(int(d)) for d in tuple(shape)) if len(tuple(shape)) else "()"


def _node_color(node):
    """按节点语义配色：参数=橙、常量=灰、损失=红、普通运算=蓝。

    判定规则：
        - op_name == 'Parameter' 或叶子且 requires_grad -> 参数（橙）；
        - 叶子且非参数 -> 常量（灰）；
        - op_name 含 loss/nll/entropy/bce/mse 等关键字 -> 损失（红）；
        - 其余 -> 普通运算（蓝）。
    """
    op = str(getattr(node, "op_name", "") or "")
    parents = getattr(node, "parents", []) or []
    op_low = op.lower()
    if any(k in op_low for k in ("loss", "nll", "entropy", "bce", "mse", "crossentropy")):
        return "#f4a6a6", "#b03030"                         # 损失节点：浅红底 + 深红边
    if op == "Parameter" or (not parents and getattr(node, "requires_grad", False)):
        return "#ffd28c", "#b8860b"                         # 参数节点：浅橙底 + 深黄边
    if not parents:                                         # 无父节点的非参数 -> 常量（输入数据）
        return "#dddddd", "#888888"                         # 常量：浅灰
    return "#bcd9f7", "#3a6ea5"                             # 普通运算：浅蓝


def draw_computation_graph(root, path, title="计算图", max_nodes=200, figsize=None):
    """把以 root 为输出的计算图画成分层 DAG 并保存为 PNG（纯 matplotlib，无 graphviz）。

    布局算法：
        1) 本地拓扑排序得到节点序列；
        2) 节点深度 = 0（叶子）或 1 + max(父节点深度)，x 坐标 = 深度
           （输入在左，root/损失在最右）；
        3) 同层节点均匀分布占据 y 轴，并按"父节点平均 y"排序两轮，
           减少边的交叉；
        4) 每个节点画圆角文本框（参数名 + 算子名 + 形状），边画箭头。

    参数：
        root (Node): 计算图输出节点；
        path (str): PNG 保存路径（自动创建父目录）；
        title (str): 图标题；
        max_nodes (int): 最大绘制节点数，超限时从 root 出发按"最近祖先"截取；
        figsize (tuple | None): 画布尺寸，None 时按图规模自适应。

    返回：
        (n_nodes, n_edges)：实际绘制的节点数与边数。
    """
    _ensure_chinese_font()                                  # 中文字体兜底（防止标题乱码）
    # ---------- 1. 拓扑排序 + 超限截断 ----------
    order = _local_topo(root)                               # 全图拓扑序
    if len(order) > max_nodes:                              # 节点过多：从 root 反向 BFS 截取
        print(f"[visualization] 警告: 节点数 {len(order)} 超过 max_nodes={max_nodes}，"
              f"仅绘制离输出最近的 {max_nodes} 个节点")
        keep, queue, seen = [], [root], {id(root)}
        while queue and len(keep) < max_nodes:              # BFS 取最近祖先
            n = queue.pop(0)
            keep.append(n)
            for p in getattr(n, "parents", []) or []:
                if id(p) not in seen:
                    seen.add(id(p))
                    queue.append(p)
        idset = {id(n) for n in keep}
        order = [n for n in _local_topo(root) if id(n) in idset]  # 保持拓扑序

    nodes = order
    edges = _collect_edges_local(nodes)                     # 收集要画的边
    nid = {id(n): i for i, n in enumerate(nodes)}           # 节点 id -> 序号

    # ---------- 2. 计算每个节点的拓扑深度 ----------
    depth = {}                                              # 节点id -> 深度
    for n in nodes:                                         # 拓扑序保证父节点先算好
        ps = [p for p in (getattr(n, "parents", []) or []) if id(p) in depth]
        depth[id(n)] = (1 + max(depth[id(p)] for p in ps)) if ps else 0

    # ---------- 3. 分层并计算 (x, y) 坐标 ----------
    layers = {}                                             # 深度 -> [节点, ...]
    for n in nodes:
        layers.setdefault(depth[id(n)], []).append(n)
    max_depth = max(layers.keys())
    pos = {}                                                # 节点id -> (x, y)

    def _assign_y(layer_nodes):
        # 同层均匀铺开：以 0 为中心，间距 1.0
        k = len(layer_nodes)
        for i, n in enumerate(layer_nodes):
            pos[id(n)] = (depth[id(n)], i - (k - 1) / 2.0)

    for d in range(max_depth + 1):                          # 第一遍：按原顺序铺开
        _assign_y(layers.get(d, []))
    for _ in range(2):                                      # 两遍微调：父节点均值排序减少交叉
        for d in range(max_depth + 1):
            layer = layers.get(d, [])
            if len(layer) < 2:
                continue
            # 排序键 = 各父节点 y 的均值（无父节点用当前 y）
            def key(n):
                ps = [p for p in (getattr(n, "parents", []) or []) if id(p) in pos]
                return (np.mean([pos[id(p)][1] for p in ps])
                        if ps else pos[id(n)][1])
            layers[d] = sorted(layer, key=key)
            _assign_y(layers[d])

    # ---------- 4. 建画布（尺寸自适应） ----------
    widest = max((len(v) for v in layers.values()), default=1)  # 最宽层的节点数
    if figsize is None:                                     # 宽度随深度、高度随最宽层伸缩
        figsize = (min(40.0, max(8.0, (max_depth + 1) * 2.2)),
                   min(30.0, max(5.0, widest * 1.15 + 1.5)))
    fig, ax = plt.subplots(figsize=figsize)

    # ---------- 5. 画边（箭头从 parent 指向 child） ----------
    for p, c in edges:                                      # 逐条画箭头
        (x0, y0), (x1, y1) = pos[id(p)], pos[id(c)]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),       # 箭头终点=child 起点=parent
                    arrowprops=dict(arrowstyle="-|>", color="#7a8aa0",
                                    lw=1.0, shrinkA=14, shrinkB=14,
                                    connectionstyle="arc3,rad=0.06"))

    # ---------- 6. 画节点（圆角文本框） ----------
    fontsize = 8 if widest <= 12 else 6                     # 节点很多时缩小字号
    for n in nodes:                                         # 逐个画节点框
        x, y = pos[id(n)]
        name = str(getattr(n, "name", "") or "")            # 参数名/调试名
        op = str(getattr(n, "op_name", "") or "?")          # 产生该节点的算子名
        shape_s = _node_shape_str(n)                        # 张量形状字符串
        # 框内两行文本：第一行 名称(如有)，第二行 算子 + 形状
        line1 = name if name else op
        line2 = f"{op}: {shape_s}" if name else shape_s
        fc, ec = _node_color(n)                             # 语义配色
        is_root = (n is root) or (id(n) == id(root))
        ax.text(x, y, f"{line1}\n{line2}", ha="center", va="center",
                fontsize=fontsize, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec,
                          lw=1.6 if is_root else 1.0))      # 输出节点加粗描边

    # ---------- 7. 图例 / 标题 / 保存 ----------
    legend_items = [Patch(fc="#ffd28c", ec="#b8860b", label="参数 Parameter"),
                    Patch(fc="#dddddd", ec="#888888", label="常量/输入 Constant"),
                    Patch(fc="#bcd9f7", ec="#3a6ea5", label="运算 Op"),
                    Patch(fc="#f4a6a6", ec="#b03030", label="损失 Loss")]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(f"{title}（节点 {len(nodes)} 个，边 {len(edges)} 条）", fontsize=12)
    ax.set_xlabel("拓扑深度（左=输入，右=输出）")            # x 轴语义说明
    ax.set_yticks([])                                       # y 只用于分层排布，无实际含义
    # 关键：本图只用 text/annotate 绘制，没有参与自动缩放的 Line2D/PathPatch，
    # matplotlib 会把坐标范围留成默认的 (-0.055, 0.055)，导致所有文本落在视口外，
    # bbox_inches="tight" 为了包含它们会把画布撑到上百英寸（PNG 高达上万像素）。
    # 因此必须根据节点坐标显式设定坐标范围。
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)               # 左右各留 1 个节点的余量
    ax.set_ylim(min(ys) - 0.9, max(ys) + 1.4)               # 顶部多留空间给图例
    ax.set_aspect('auto')

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)  # 保证输出目录存在
    fig.savefig(path, dpi=150, bbox_inches="tight")         # 保存 PNG
    plt.close(fig)                                          # 关闭 figure 释放内存（Agg 后端）
    return len(nodes), len(edges)


# ======================================================================
# 三、通用训练曲线画图
# ======================================================================
def plot_loss_curve(history, path, title="训练曲线"):
    """绘制训练过程曲线：损失（左轴）+ 准确率（右轴）双 y 轴。

    history 形如 {'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}，
    采用"有啥画啥"策略：缺哪个键就少画哪条曲线。

    参数：
        history (dict): Trainer.fit 返回的历史记录字典；
        path (str): PNG 保存路径；
        title (str): 图标题。
    """
    _ensure_chinese_font()                                  # 中文字体兜底（防止标题乱码）
    fig, ax1 = plt.subplots(figsize=(8, 5))                 # 左轴画损失
    has_loss = False
    if history.get("train_loss"):                           # 训练损失（左轴，蓝实线）
        ax1.plot(history["train_loss"], "-o", ms=3, color="#1f77b4", label="训练损失")
        has_loss = True
    if history.get("val_loss"):                             # 验证损失（左轴，橙实线）
        ax1.plot(history["val_loss"], "-s", ms=3, color="#ff7f0e", label="验证损失")
        has_loss = True
    if has_loss:
        ax1.set_xlabel("轮次 Epoch")
        ax1.set_ylabel("损失 Loss")
        ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()                                       # 右轴画准确率
    has_acc = False
    if history.get("val_acc"):                              # 验证准确率（右轴，绿虚线）
        ax2.plot(history["val_acc"], "--^", ms=3, color="#2ca02c", label="验证准确率")
        has_acc = True
    if history.get("train_acc"):                            # 训练准确率（如有）
        ax2.plot(history["train_acc"], ":", color="#9467bd", label="训练准确率")
        has_acc = True
    if has_acc:
        ax2.set_ylabel("准确率 Accuracy")

    # 合并双轴图例（去重后放到不遮挡数据的位置）
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    ax1.set_title(title)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")         # 保存并关闭，禁止 show
    plt.close(fig)


def plot_lines(curves, xlabel, ylabel, title, path):
    """通用多曲线对比图：curves 的每个键是一条曲线，键名作图例。

    参数：
        curves (dict[str, list]): {曲线名: y 值列表}；
        xlabel (str): x 轴标签；
        ylabel (str): y 轴标签；
        title (str): 图标题；
        path (str): PNG 保存路径。
    """
    _ensure_chinese_font()                                  # 中文字体兜底（防止标题乱码）
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, ys in curves.items():                         # 逐条绘制，键名做图例
        ax.plot(ys, "-o", ms=3, label=str(name))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if curves:
        ax.legend(fontsize=9)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_images_grid(images, titles=None, path=None, n=8, cols=4):
    """把一批图像排成网格展示（灰度图），每张图下方显示对应标题。

    兼容三种输入形状：
        (N,H,W) 灰度 / (N,1,H,W) 单通道 / (N,3,H,W) RGB（转置后彩色显示）。

    参数：
        images (np.ndarray): 图像数组；
        titles (list[str] | None): 每张图下方的小标题；
        path (str | None): PNG 保存路径；None 则只建图不保存；
        n (int): 最多展示的图像张数（默认 8）；
        cols (int): 每行列数（默认 4）。
    """
    _ensure_chinese_font()                                  # 中文字体兜底（防止标题乱码）
    imgs = np.asarray(images)
    if imgs.ndim == 4 and imgs.shape[1] == 1:               # (N,1,H,W) -> (N,H,W)
        imgs = imgs[:, 0]
    elif imgs.ndim == 4 and imgs.shape[1] == 3:             # (N,3,H,W) -> (N,H,W,3) 彩色
        imgs = imgs.transpose(0, 2, 3, 1)
    n_show = min(n, len(imgs))                              # 实际展示张数
    rows = math.ceil(n_show / cols)                         # 行数 = 向上取整
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.4))
    axes = np.atleast_2d(axes)                              # 统一成二维索引
    for i in range(rows * cols):                            # 遍历所有格子
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")                                      # 隐藏坐标轴
        if i < n_show:
            # 灰度 cmap 显示；0~1 float 直接显示即可
            ax.imshow(imgs[i], cmap="gray")
            if titles is not None and i < len(titles):
                # 注意：ax.axis("off") 会把 xlabel 一并隐藏，所以标题必须用
                # set_title（标题不属于坐标轴装饰，关闭坐标轴后仍会显示）。
                ax.set_title(str(titles[i]), fontsize=8, pad=3)
    if path is not None:                                    # 需要保存时
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)                                          # 关闭释放内存
    return fig
