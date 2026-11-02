# -*- coding: utf-8 -*-
"""
experiments/exp4_preprocess_demo.py —— 扩展④：数据预处理与增强演示
======================================================================

对 miniflows/data.py 的核心预处理功能做端到端演示，每个小节出一张图：

1. 鸢尾花标准化前后箱线图对比   —— 观察 z-score 把各特征拉到同一量纲；
2. one-hot 编码示例表           —— 整数标签 -> 0/1 矩阵的可视化表格；
3. train_test_split 分层抽样前后类别分布 —— 分层保证训练/测试集类别比例一致；
4. MNIST 归一化前后像素统计     —— 均值/方差/最大值对比；
5. random_shift_images 平移增强前后图像对比网格。

运行方式（项目根目录）：python experiments/exp4_preprocess_demo.py
输出目录：outputs/exp4/
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
import matplotlib.pyplot as plt

from miniflows.data import (standard_scale, one_hot, train_test_split,
                            load_iris, load_mnist_arrays, random_shift_images)
from miniflows.visualization import plot_images_grid

OUT_DIR = os.path.join(_ROOT, "outputs", "exp4")              # 契约第 12 节约定的输出目录
os.makedirs(OUT_DIR, exist_ok=True)                           # 自动创建输出目录
FEATURE_NAMES = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]  # 鸢尾花 4 特征中文名
conclusions = []                                              # 收集各小节结论，最后写 md


# ======================================================================
# 小节 1：鸢尾花标准化前后箱线图对比
# ======================================================================
def sec1_iris_standardize():
    """演示 z-score 标准化：把量纲差异巨大的 4 个特征拉到均值 0 方差 1。"""
    print("\n" + "=" * 60)
    print("小节 1：鸢尾花标准化前后箱线图对比")
    print("=" * 60)
    X, y, class_names = load_iris()                          # 加载鸢尾花（150,4）
    X_scaled, scaler = standard_scale(X)                     # z-score 标准化
    print(f"原始数据:   均值 = {np.round(X.mean(axis=0), 2)}")
    print(f"           标准差 = {np.round(X.std(axis=0), 2)}")
    print(f"标准化后:   均值 = {np.round(X_scaled.mean(axis=0), 6)}")
    print(f"           标准差 = {np.round(X_scaled.std(axis=0), 6)}")

    # ---------- 并排箱线图：左原始右标准化 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, data, name in ((axes[0], X, "标准化前（原始量纲）"),
                           (axes[1], X_scaled, "标准化后（均值0/方差1）")):
        bp = ax.boxplot(data, patch_artist=True, tick_labels=FEATURE_NAMES)
        for box in bp["boxes"]:                              # 箱子上淡蓝色
            box.set(facecolor="#cfe3f7")
        ax.set_title(name)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("鸢尾花特征标准化前后箱线图对比（扩展④）")
    path = os.path.join(OUT_DIR, "箱线图_标准化对比.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)                                           # 保存后关闭，禁止 show
    print(f"[已保存] {path}")
    # 记录结论（供最后 md 汇总）
    conclusions.append(
        "1. 标准化（z-score）：原始 4 个特征量纲差异大（花萼长度均值约 5.8、"
        "花瓣宽度约 1.2），标准化后各列均值≈0、标准差≈1，箱线图被拉到同一尺度；"
        "这能避免大数值特征在梯度更新中主导训练。apply_scale 可用同一组统计量"
        "安全地变换测试集，避免数据泄露。")
    return X, y, class_names, X_scaled


# ======================================================================
# 小节 2：one-hot 编码示例表
# ======================================================================
def sec2_one_hot_table(y, class_names):
    """展示 one-hot 编码：整数标签 -> (N,C) 0/1 矩阵，画成示例表格。"""
    print("\n" + "=" * 60)
    print("小节 2：one-hot 编码示例表")
    print("=" * 60)
    n_show, C = 8, len(class_names)                          # 展示前 8 个样本
    y_head = y[:n_show]                                      # 前 8 个整数标签
    oh = one_hot(y_head, num_classes=C)                      # one-hot 矩阵 (8,3)
    # ---------- 控制台打印文本表格 ----------
    print(f"{'样本':>4} | {'整数标签':>6} | {class_names}")
    for i in range(n_show):
        print(f"{i:>4} | {int(y_head[i]):>6}   | {oh[i].astype(int)}")

    # ---------- matplotlib 表格图 ----------
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")                                           # 隐藏坐标轴只留表格
    col_labels = ["样本编号", "整数标签"] + class_names       # 表头
    cell = [[str(i), str(int(y_head[i]))] + [f"{int(v)}" for v in oh[i]]
            for i in range(n_show)]                          # 表体
    tab = ax.table(cellText=cell, colLabels=col_labels, loc="center",
                   cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.0, 1.4)                                      # 行高放大更清晰
    # 整数标签为 1 的 one-hot 单元格高亮浅橙色
    for (r, c), cell_ in tab.get_celld().items():
        if r > 0 and c >= 2 and cell[c - 2] == "1":          # 命中类别列
            cell_.set_facecolor("#ffd28c")
    ax.set_title("one-hot 编码示例表：整数标签 -> 0/1 向量")
    path = os.path.join(OUT_DIR, "one_hot编码表.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] {path}")
    conclusions.append(
        "2. one-hot 编码：整数标签 0/1/2 变成长度为 3 的 0/1 向量，"
        "命中类别置 1（表中橙色高亮）。配合交叉熵/均方误差使用，"
        "使\"类别间无序\"这一先验显式化。")
    return oh


# ======================================================================
# 小节 3：train_test_split 分层抽样前后类别分布
# ======================================================================
def sec3_stratified_split(X, y, class_names):
    """对比 原始分布 / 随机划分测试集 / 分层划分测试集 的类别比例。"""
    print("\n" + "=" * 60)
    print("小节 3：train_test_split 分层抽样前后类别分布")
    print("=" * 60)
    test_size = 0.3                                          # 测试集比例 30%
    _, _, X_te_rand, y_te_rand = train_test_split(           # 不分层：纯随机划分
        X, y, test_size=test_size, shuffle=True, stratify=None, seed=42)
    _, _, X_te_str, y_te_str = train_test_split(             # 分层：按类别比例划分
        X, y, test_size=test_size, shuffle=True, stratify=y, seed=42)
    C = len(class_names)
    # 各数据集的类别占比（归一化直方图）
    stats = {
        "原始数据集": np.bincount(y, minlength=C) / len(y),
        "随机划分-测试集": np.bincount(y_te_rand, minlength=C) / len(y_te_rand),
        "分层划分-测试集": np.bincount(y_te_str, minlength=C) / len(y_te_str),
    }
    for name, prop in stats.items():                         # 控制台打印占比
        print(f"{name:<10}: " + "  ".join(
            f"{class_names[i]}={prop[i]:.3f}" for i in range(C)))

    # ---------- 分组柱状图 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25                                             # 每根柱子宽度
    xs = np.arange(C)                                        # 每类一组
    colors = ["#8fb9de", "#f4b183", "#a9d18e"]               # 三组颜色
    for i, (name, prop) in enumerate(stats.items()):
        ax.bar(xs + (i - 1) * width, prop, width, label=name, color=colors[i])
    ax.set_xticks(xs)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("类别占比")
    ax.set_title(f"train_test_split 分层抽样前后类别分布对比（test_size={test_size}）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    path = os.path.join(OUT_DIR, "类别分布_分层抽样.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] {path}")
    conclusions.append(
        "3. 分层抽样（stratify=y）：随机划分的测试集类别占比会有波动"
        "（小类别可能被低估），分层划分让测试集各类占比与原始数据集几乎一致"
        "（柱高对齐），小样本任务中能显著降低评估偏差。")
    return y_te_str


# ======================================================================
# 小节 4：MNIST 归一化前后像素统计
# ======================================================================
def sec4_mnist_stats():
    """对比 MNIST 原始像素 [0,255] 与归一化后 [0,1] 的均值/方差/最大值。"""
    print("\n" + "=" * 60)
    print("小节 4：MNIST 归一化前后像素统计")
    print("=" * 60)
    try:
        # 子集 2000 张（按类别均衡取样）足以估计统计量，且首次下载后自动缓存
        Xtr, ytr, _, _ = load_mnist_arrays(subset_train=2000, subset_test=None,
                                           data_dir=os.path.join(_ROOT, "data"))
    except RuntimeError as e:                                # 下载彻底失败：跳过本节
        print(f"[警告] MNIST 加载失败，跳过本小节。\n{e}")
        conclusions.append("4. MNIST 归一化：数据下载失败，本小节跳过。")
        return None
    raw = Xtr * 255.0                                        # 还原到 [0,255] 作为"归一化前"

    # ---------- 计算并打印统计量 ----------
    def _stats(imgs, tag):
        s = {"均值": float(imgs.mean()), "方差": float(imgs.var()),
             "最大值": float(imgs.max()), "最小值": float(imgs.min())}
        print(f"{tag}: " + "  ".join(f"{k}={v:.4f}" for k, v in s.items()))
        return s
    s_raw = _stats(raw, "归一化前 [0,255]")
    s_norm = _stats(Xtr, "归一化后 [0,1]  ")

    # ---------- 统计对比图（柱状表 + 样例图） ----------
    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(1, 2, 1)                           # 左：统计量柱状图
    keys = ["均值", "方差", "最大值"]
    # 归一化前后同画一轴：原始值除以 255 换算到 [0,1] 尺度便于对比
    raw_vals = [s_raw[k] / 255.0 for k in keys]
    norm_vals = [s_norm[k] for k in keys]
    xs = np.arange(len(keys))
    ax1.bar(xs - 0.18, raw_vals, 0.36, label="归一化前（÷255 折算）", color="#f4b183")
    ax1.bar(xs + 0.18, norm_vals, 0.36, label="归一化后", color="#8fb9de")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(keys)
    ax1.set_ylabel("像素统计量（[0,1] 尺度）")
    ax1.set_title("MNIST 像素统计：归一化前后")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")
    ax2 = fig.add_subplot(1, 2, 2)                           # 右：样例数字
    ax2.imshow(Xtr[0, 0], cmap="gray")
    ax2.axis("off")
    ax2.set_title(f"样例（标签 {int(ytr[0])}，已归一化）")
    fig.suptitle("MNIST 归一化前后像素统计（均值/方差/最大值）")
    path = os.path.join(OUT_DIR, "MNIST像素统计.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] {path}")
    conclusions.append(
        f"4. MNIST 归一化：像素从 [0,255] 线性缩放到 [0,1]"
        f"（均值 {s_raw['均值']:.2f}→{s_norm['均值']:.4f}，最大值 {s_raw['最大值']:.0f}→"
        f"{s_norm['最大值']:.1f}）。小数值输入让 sigmoid/softmax 等的指数运算"
        "远离溢出区，也使学习率跨数据集可迁移。load_mnist_arrays 支持按类别"
        "均衡取样子集，方便快速实验。")
    return Xtr


# ======================================================================
# 小节 5：random_shift_images 平移增强前后对比
# ======================================================================
def sec5_random_shift(Xtr):
    """展示平移增强：同一批数字随机循环平移后的对比网格。"""
    print("\n" + "=" * 60)
    print("小节 5：random_shift_images 平移增强前后对比")
    print("=" * 60)
    if Xtr is None:                                          # MNIST 不可用则跳过
        print("[警告] 无 MNIST 数据，跳过本小节")
        return
    n = 8                                                    # 展示 8 张
    X8 = Xtr[:n, 0]                                          # (8,28,28)
    X8_shift = random_shift_images(X8, max_shift=2, seed=0)  # 平移增强（±2 像素循环平移）
    # 定量对比：非零像素质心变化，验证图像确实被平移
    def _centroid(img):
        ys, xs_ = np.nonzero(img > 0.05)                     # 显著非零像素坐标
        return (ys.mean(), xs_.mean()) if len(ys) else (np.nan, np.nan)
    shifts = [np.linalg.norm(np.array(_centroid(b)) - np.array(_centroid(a)))
              for a, b in zip(X8, X8_shift)]                 # 增强前后质心位移
    print(f"平移前后质心平均位移: {np.mean(shifts):.2f} 像素（max_shift=2）")

    # ---------- 对比网格：上排原始 / 下排增强 ----------
    imgs = np.concatenate([X8, X8_shift], axis=0)            # 16 张
    titles = ([f"原始{i}" for i in range(n)]
              + [f"平移后{i}" for i in range(n)])
    path = os.path.join(OUT_DIR, "增强对比_平移.png")
    plot_images_grid(imgs, titles=titles, path=path, n=2 * n, cols=8)
    print(f"[已保存] {path}")
    conclusions.append(
        "5. 随机平移增强（random_shift_images）：对每张图沿高度/宽度方向"
        f"随机整数循环平移（±2 像素），上排原图下排增强图肉眼可见笔画位置变化"
        f"（质心平均位移 {np.mean(shifts):.2f} 像素）。等价于免费扩充训练集，"
        "提升 CNN 对书写位置偏移的鲁棒性、抑制过拟合。")


# ======================================================================
# 主流程
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("扩展④：数据预处理与增强演示（miniflows/data.py）")
    print("=" * 60)

    # ---------- 小节 1~3：鸢尾花 ----------
    X, y, class_names, X_scaled = sec1_iris_standardize()    # 标准化 + 箱线图
    sec2_one_hot_table(y, class_names)                       # one-hot 表格
    sec3_stratified_split(X, y, class_names)                 # 分层抽样分布

    # ---------- 小节 4~5：MNIST ----------
    Xtr = sec4_mnist_stats()                                 # 归一化统计
    sec5_random_shift(Xtr)                                   # 平移增强对比

    # ---------- 汇总结论 md ----------
    md_path = os.path.join(OUT_DIR, "预处理演示结论.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 扩展④ 数据预处理与增强演示结论\n\n")
        f.write("本报告由 experiments/exp4_preprocess_demo.py 自动生成。\n\n")
        for line in conclusions:                             # 各小节结论逐条写入
            f.write(line + "\n\n")
    print(f"\n[已保存] {md_path}")
    print("\n" + "=" * 60)
    print("扩展④ 数据预处理演示完成！全部图片输出至 outputs/exp4/")
    print("=" * 60)
