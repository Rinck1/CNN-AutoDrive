# `miniflows/visualization.py` 阅读指南

> 前置：Node、matplotlib ｜ 难度：★★ ｜ 建议 45 分钟 ｜ 扩展①

## 1. 文件定位

纯 matplotlib 的可视化工具：

- 计算图 DAG（不依赖 graphviz）；
- 训练损失/准确率曲线；
- 通用多曲线对比；
- 图像网格（带标题）。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `setup_chinese_font` (39) / `_ensure_chinese_font` (51) | 中文字体兜底 |
| 2 | `_local_topo` (65) / `_collect_edges_local` (91) | 图遍历 |
| 3 | `_node_shape_str` (112) / `_node_color` (123) | 节点文字/配色 |
| 4 | `draw_computation_graph` (144) | 分层 DAG 绘图 |
| 5 | `plot_loss_curve` (280) | loss 左轴 + acc 右轴 |
| 6 | `plot_lines` (327) | 多曲线对比 |
| 7 | `plot_images_grid` (352) | 图像网格 |

## 3. 逐段讲解

### 3.1 计算图布局

1. 从 root 做拓扑排序；
2. 每个节点深度 = 1 + max(父节点深度)，叶子深度 0；
3. 同层节点按 y 均匀分布；再按“父节点平均 y”排序两轮，减少交叉；
4. x = 深度，y = 层内位置；
5. 节点画圆角框（名称/算子/形状），边画箭头（父→子）。

节点配色：参数橙、常量灰、运算蓝、损失红。

### 3.2 两个必须知道的坑

**（1）坐标轴范围**：图中只有 text/annotation，没有参与 autoscale 的 Line2D/PathPatch，
matplotlib 会把 xlim/ylim 留成 (-0.055, 0.055)，所有文字落在视口外，
`bbox_inches='tight'` 会把画布撑到上百英寸（曾生成 98645×30517 的 PNG）。
解决：根据节点坐标显式 `set_xlim/set_ylim`。

**（2）节点太多**：`max_nodes=200`，超限时从 root 反向 BFS 保留最近祖先，
避免“毛线团”。

### 3.3 曲线

`plot_loss_curve(history, path)`：

- 左轴 train/val loss；
- 右轴 val_acc（若存在）；
- 无 `plt.show()`，保存后 `plt.close`。

`plot_lines(curves, xlabel, ylabel, title, path)`：字典 `{图例: 序列}` 的多曲线。

### 3.4 图像网格

- 支持 `(N,H,W)`、`(N,1,H,W)`、`(N,3,H,W)`；
- 灰度图用 `cmap='gray'`；
- **标题必须用 `set_title`**：`ax.axis('off')` 会连 xlabel 一起隐藏；
- `path=None` 时只建图不保存。

## 4. 调用关系

exp1 用计算图；所有训练脚本用 `plot_loss_curve`；exp4/DonkeyCar 用图像网格；
ViT 的注意力图是应用脚本里单独画的，不在此模块。

## 5. 易错点

1. 坐标轴范围不显式设置 → 巨大 PNG；
2. `axis('off')` 隐藏标题；
3. 中文字体缺失 → 方块（Linux 需装字体）；
4. 每张图记得 `plt.close(fig)`，否则循环中内存增长；
5. `bbox_inches='tight'` 可裁边，但前提是坐标范围正确。

## 6. 自检问题

1. 计算图的深度如何计算？为什么按深度分层能减少交叉？
2. 不设置 xlim/ylim 会发生什么？为什么？
3. 图像网格标题为什么不能用 xlabel？
4. `plot_loss_curve` 的双轴分别是什么？
