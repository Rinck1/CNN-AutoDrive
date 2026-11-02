# -*- coding: utf-8 -*-
"""
logistic_regression.py —— 基于 MiniFlows 的二分类逻辑回归应用（课程基本任务③）
==============================================================================

功能概述：
    1. 用 miniflows.data.make_blobs 生成两团二维高斯簇的二分类数据（n=300）；
    2. 模型为单个线性层 Linear(2,1)，输出 logits；配合 BCEWithLogitsLoss
       （数值稳定的二分类交叉熵，内部含 sigmoid，无需显式激活）；
    3. 用 Adam（lr=0.05）训练 200 轮；
    4. 打印测试集准确率（两团高斯簇线性可分，应 ≥95%）与学到的决策边界参数；
    5. 保存产物：
       - outputs/logistic_regression/决策边界.png（散点 + 网格预测等高线画出的边界）
       - outputs/logistic_regression/损失曲线.png（训练/验证损失与验证准确率）

实现要点（行内注释见代码）：
    - 决策边界：对输入平面上密集网格点逐点 predict 得到 logits，
      logits=0（即 sigmoid 概率=0.5）处即为线性决策边界 w1*x1 + w2*x2 + b = 0，
      用 matplotlib 的 contour/contourf 可视化。

运行方式（在项目根目录）：
    python applications/logistic_regression.py
"""

# ==================== 标准脚本初始化头（契约第 1 节，原样复制） ====================
import os
import sys
try:  # Windows 控制台强制 UTF-8，避免中文乱码报错
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # 使脚本能 import miniflows / common_setup
from common_setup import init_env
init_env(seed=42)  # 统一设置：随机种子 + matplotlib 中文字体 + Agg 后端
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from miniflows.layers import Linear, Sequential   # 线性层与顺序容器
from miniflows.losses import BCEWithLogitsLoss    # 数值稳定二分类交叉熵
from miniflows.optim import Adam                  # Adam 优化器
from miniflows.trainer import Trainer             # 统一训练器
from miniflows.data import make_blobs, train_test_split       # 合成数据与划分
from miniflows.visualization import plot_loss_curve           # 训练曲线图

# ------------------------- 全局常量（超参数集中管理） -------------------------
OUT_DIR = init_env(seed=42, output_dir='outputs/logistic_regression')  # 输出目录
N_SAMPLES = 300      # 样本总数（两团各约 150）
N_FEATURES = 2       # 特征维度：平面上的 (x1, x2)
EPOCHS = 200         # 训练轮数：逻辑回归凸问题，200 轮充分收敛
BATCH_SIZE = 32      # 小批量大小：经典取值
LR = 0.05            # Adam 学习率：题目指定，逻辑回归上收敛快且稳
TEST_SIZE = 0.3      # 测试集占比 30%，留 90 个点评估泛化


def build_model():
    """构建逻辑回归模型 Linear(2,1)。

    返回：
        Sequential: 单个 Linear(2,1) 层；输出 (N,1) 的 logits（未过 sigmoid），
                    由 BCEWithLogitsLoss 在损失内部做数值稳定 sigmoid。
    """
    # 输入特征 2 维 -> 输出 1 维 logits；正类判据：logits > 0
    return Sequential(Linear(2, 1))


def plot_decision_boundary(trainer, X, y, path):
    """绘制数据散点与线性决策边界（用网格点预测 + 等高线实现）。

    实现思路：
        1) 在特征平面上取 300x300 的密集网格；
        2) 用训练好的模型对每个网格点预测 logits；
        3) logits 过 sigmoid 得到 P(类别=1) 的概率图；
        4) contourf 填充两类区域，contour(level=0.5) 画出概率 0.5 等值线
           ——即决策边界。

    参数：
        trainer (Trainer): 已训练的 Trainer（predict 返回原始 logits）。
        X (np.ndarray): 全部样本特征 (N,2)。
        y (np.ndarray): 全部样本标签 (N,)，取值 {0,1}。
        path (str): 图片保存路径。
    """
    # 网格范围：在数据最小/最大值基础上外扩 1.0，留出边距更好看
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    # 生成 300x300 网格坐标（分辨率够高，边界平滑）
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])   # (90000,2) 网格点

    # 对全部网格点批量预测 logits（batch 放大以减少循环次数）
    logits = trainer.predict(grid, batch_size=16384).reshape(xx.shape)  # (300,300)
    # sigmoid 概率：P(y=1)；数值稳定写法，避免 exp 溢出
    prob = 1.0 / (1.0 + np.exp(-logits))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    # 填充两类预测区域：概率>0.5 判为类别 1（橙色区域），否则类别 0（蓝色区域）
    ax.contourf(xx, yy, prob, levels=[0.0, 0.5, 1.0],
                colors=['#aec7e8', '#ffbb78'], alpha=0.45)
    # 概率=0.5 的等值线即决策边界（黑色实线）
    ax.contour(xx, yy, prob, levels=[0.5], colors='k', linewidths=2)
    # 叠加画真实样本散点：两类不同颜色，便于观察分错点
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='tab:blue', s=22,
               edgecolors='k', linewidths=0.4, label='类别 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='tab:orange', s=22,
               edgecolors='k', linewidths=0.4, label='类别 1')
    ax.set_xlabel('特征 x1')
    ax.set_ylabel('特征 x2')
    ax.set_title('逻辑回归决策边界（MiniFlows，make_blobs 二分类）')
    ax.legend(loc='upper right')
    fig.savefig(path, dpi=150, bbox_inches='tight')  # 只保存，禁止 plt.show()
    plt.close(fig)


# ================================ 主流程 ================================
def main():
    """逻辑回归主流程：造数据 -> 训练 -> 评估 -> 打印结论 -> 保存图表。"""
    print('=' * 60)
    print('MiniFlows 逻辑回归：make_blobs 两团高斯二分类（n=%d）' % N_SAMPLES)
    print('=' * 60)

    # ---------- 1. 生成两团高斯簇数据 ----------
    # centers=2 -> 两团；std=1.0 -> 团内标准差，两团相距足够远、近似线性可分
    X, y = make_blobs(n_samples=N_SAMPLES, centers=2,
                      center_box=(-3.0, 3.0), std=1.0,
                      n_features=N_FEATURES, seed=42)
    print(f'数据生成完成：X.shape={X.shape}, y 分布：'
          f'类0={int((y == 0).sum())} 个, 类1={int((y == 1).sum())} 个')

    # ---------- 2. 划分训练/测试集（分层抽样保持两类比例） ----------
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              shuffle=True, stratify=y, seed=42)
    print(f'训练集 {X_tr.shape[0]} 条，测试集 {X_te.shape[0]} 条（分层抽样）')

    # ---------- 3. 构建模型 / 损失 / 优化器 / 训练器 ----------
    model = build_model()
    # BCEWithLogitsLoss：输入 logits，内部做数值稳定 sigmoid + 交叉熵
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.params(), lr=LR)
    # task='classification'：evaluate 时输出维度为 1 会按 logits>0.5? 
    # ——按契约，单输出配合 BCEWithLogitsLoss 时以 >0.5 判别（Trainer 内部处理）
    trainer = Trainer(model, loss_fn, optimizer, task='classification')

    # ---------- 4. 训练（把测试集作为验证集，逐轮记录 val_loss / val_acc） ----------
    # verbose=False：200 轮逐轮打印冗长，只保留关键汇总
    history = trainer.train(X_tr, y_tr, X_val=X_te, y_val=y_te,
                          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=False)

    # ---------- 5. 测试集评估（Trainer 内部按 logits>0 判 1，等价 sigmoid>0.5） ----------
    test_loss, test_acc = trainer.evaluate(X_te, y_te, batch_size=256)
    print('-' * 60)
    print(f'最终训练损失: {history["train_loss"][-1]:.6f}')
    print(f'测试集平均损失: {test_loss:.6f}')
    print(f'测试集准确率: {test_acc * 100:.2f}%')

    # 提取决策边界参数 w1, w2, b 并打印边界方程
    w1 = w2 = b = None
    for name, p in model.named_params():  # named_params -> [('layer0.W', Node), ...]
        if name.endswith('.W'):
            w1, w2 = float(p.data[0, 0]), float(p.data[0, 1])  # W 形状 (1,2)
        elif name.endswith('.b'):
            b = float(p.data.reshape(-1)[0])                   # b 形状 (1,)
    print(f'决策边界方程: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0')

    # ---------- 6. 中文结论 ----------
    print('【结论】')
    print(f'  两团高斯簇近似线性可分，逻辑回归测试准确率 {test_acc * 100:.2f}%，'
          f'达到预期（≥95%）。')
    print('  说明单层线性模型 + BCEWithLogitsLoss + Adam 的组合在 MiniFlows 上'
          '能正确完成二分类任务，框架的自动微分与优化器工作正常。')

    # ---------- 7. 保存图 1：决策边界图 ----------
    path1 = os.path.join(OUT_DIR, '决策边界.png')
    plot_decision_boundary(trainer, X, y, path1)
    print(f'已保存决策边界图：{path1}')

    # ---------- 8. 保存图 2：训练曲线（train/val loss + val_acc 双轴） ----------
    path2 = os.path.join(OUT_DIR, '损失曲线.png')
    plot_loss_curve(history, path2, title='逻辑回归训练曲线（Adam, lr=0.05）')
    print(f'已保存损失曲线图：{path2}')
    print('逻辑回归应用运行完毕。')


if __name__ == '__main__':
    main()
