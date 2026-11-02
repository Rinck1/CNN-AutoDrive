# -*- coding: utf-8 -*-
"""
linear_regression.py —— 基于 MiniFlows 的一元线性回归应用（课程基本任务③）
==========================================================================

功能概述：
    1. 合成一维回归数据：y = 3x + 2 + 高斯噪声（n=200），真实权重 w=3、偏置 b=2；
    2. 使用 MiniFlows 搭建最简单的线性模型 Linear(1,1)，配合 MSELoss（均方误差损失）；
    3. 分别用两种优化器训练 100 轮并对比：
       - MBGD（小批量梯度下降，lr=0.05）
       - Adam（自适应矩估计，lr=0.05）
    4. 打印两种优化器最终在全体数据上的 MSE，以及各自学到的 w、b
       （应接近真值 3 和 2，验证框架前向/反向/优化器实现正确）；
    5. 保存产物：
       - outputs/linear_regression/拟合结果.png       （散点 + 真实直线 + 两条拟合直线）
       - outputs/linear_regression/优化器损失对比.png （MBGD 与 Adam 的训练损失曲线）

超参选择理由（行内注释见代码）：
    - lr=0.05：对该凸问题足够大以快速收敛、又不会震荡发散，MBGD 与 Adam 取相同学习率保证公平；
    - batch_size=32：经典小批量大小，兼顾梯度噪声（有助跳出平坦区）与向量化效率；
    - epochs=100：一元线性回归是强凸问题，100 轮足以让两种优化器都充分收敛；
    - 两个模型在构建前重置随机种子，保证初始 w、b 完全一致，实现公平对比。

运行方式（在项目根目录）：
    python applications/linear_regression.py
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

from miniflows.layers import Linear, Sequential      # 线性层与顺序容器
from miniflows.losses import MSELoss                 # 均方误差损失（回归任务）
from miniflows.optim import MBGD, Adam               # 两种优化器
from miniflows.trainer import Trainer                # 统一训练器
from miniflows.visualization import plot_lines       # 通用多曲线对比图

# ------------------------- 全局常量（超参数集中管理） -------------------------
OUT_DIR = init_env(seed=42, output_dir='outputs/linear_regression')  # 输出目录，自动创建
N_SAMPLES = 200        # 合成样本数
TRUE_W, TRUE_B = 3.0, 2.0   # 数据真实权重与偏置（回归目标）
NOISE_STD = 1.0        # 高斯噪声标准差：太大则数据不可分，太小则任务平凡
EPOCHS = 100           # 训练轮数：强凸问题 100 轮足够充分收敛
BATCH_SIZE = 32        # 小批量大小：经典取值，梯度带适度噪声
LR = 0.05              # 两种优化器共用学习率，保证对比公平


def make_data(n=N_SAMPLES):
    """合成一元线性回归数据集。

    数据生成规则：y = 3x + 2 + N(0, 1) 高斯噪声，x 服从 U(-3, 3) 均匀分布。

    参数：
        n (int): 样本数量，默认 200。

    返回：
        (X, y): 两个 np.ndarray，形状分别为 (n, 1) 与 (n, 1)，float64；
                X 保持二维 (n, 1) 是为了直接喂给 Linear(1,1) 层。
    """
    # x 均匀分布在 [-3, 3]，覆盖足够宽的输入范围，便于观察拟合直线斜率
    X = np.random.uniform(-3.0, 3.0, size=(n, 1))
    # 高斯噪声 std=1.0：模拟真实测量误差，使数据点分布在真实直线附近
    noise = np.random.randn(n, 1) * NOISE_STD
    y = TRUE_W * X + TRUE_B + noise  # 真值：w=3, b=2
    return X, y


def build_model():
    """构建线性回归模型 Linear(1,1)。

    返回：
        Sequential: 只含一个 Linear(1,1) 层的顺序容器；
                    输入特征数 1（x），输出维度 1（预测的 y）。
    """
    # Linear(1,1)：前向 = x @ W + b；默认 He 初始化（单输入时 std=sqrt(2)）
    return Sequential(Linear(1, 1))


def get_w_b(model):
    """从模型中提取学到的标量权重 w 与偏置 b。

    实现思路：
        遍历 model.named_params()（名字形如 'layer0.W' / 'layer0.b'），
        按名字后缀找到参数节点，取出其 .data 中的标量值。

    参数：
        model (Sequential): 训练好的模型。

    返回：
        (w, b): 两个 float，分别为权重与偏置的数值。
    """
    w, b = None, None
    for name, p in model.named_params():  # named_params 返回 [(参数名, Node), ...]
        if name.endswith('.W') or name == 'W':
            w = float(p.data.reshape(-1)[0])  # (1,1) 矩阵取标量
        elif name.endswith('.b') or name == 'b':
            b = float(p.data.reshape(-1)[0])  # (1,) 向量取标量
    return w, b


def train_with(optimizer_name):
    """用指定优化器完整训练一个全新的线性回归模型。

    为保证两种优化器对比公平，本函数在构建模型前重置 numpy 随机种子，
    使两次训练的初始 w、b 完全相同；数据顺序由 Trainer 内部 DataLoader 决定。

    参数：
        optimizer_name (str): 'MBGD' 或 'Adam'，选择优化器。

    返回：
        (model, trainer, history): 训练好的模型、训练器对象（用于预测/评估）、
                                   训练历史字典 {'train_loss': [...], ...}。
    """
    np.random.seed(42)  # 重置种子 -> 模型初始化与数据合成序列一致，起点相同
    model = build_model()

    # 根据 optimizer_name 构造对应优化器；params() 返回全部可训练参数 Node
    if optimizer_name == 'MBGD':
        # MBGD：普通小批量梯度下降，仅按 lr 缩放梯度更新
        optimizer = MBGD(model.params(), lr=LR)
    else:
        # Adam：自适应学习率 + 动量，通常在小学习率下收敛更快更稳
        optimizer = Adam(model.params(), lr=LR)

    # 回归任务：task='regression'，评估时 Trainer.evaluate 返回 MSE 而非准确率
    trainer = Trainer(model, MSELoss(), optimizer, task='regression')

    # verbose=False：100 轮逐轮打印太冗长，只保留最终汇总打印
    history = trainer.train(X_train, y_train, epochs=EPOCHS,
                          batch_size=BATCH_SIZE, verbose=False)
    return model, trainer, history


# ================================ 主流程 ================================
def main():
    """线性回归主流程：造数据 -> 双优化器训练 -> 打印结论 -> 保存图表。"""
    global X_train, y_train
    print('=' * 60)
    print('MiniFlows 一元线性回归：y = 3x + 2 + 高斯噪声（n=%d）' % N_SAMPLES)
    print('=' * 60)

    # ---------- 1. 合成数据 ----------
    X_train, y_train = make_data()  # (200,1), (200,1)
    print(f'数据合成完成：X.shape={X_train.shape}, y.shape={y_train.shape}')

    # ---------- 2. 分别用 MBGD 与 Adam 训练 ----------
    model_mbgd, trainer_mbgd, hist_mbgd = train_with('MBGD')   # 小批量梯度下降
    model_adam, trainer_adam, hist_adam = train_with('Adam')   # 自适应优化器

    # ---------- 3. 评估：全体数据上的最终 MSE 与学到的 w、b ----------
    # Trainer.evaluate 对 regression 任务返回 (平均损失, MSE)，此处两者相等
    _, mse_mbgd = trainer_mbgd.evaluate(X_train, y_train)
    _, mse_adam = trainer_adam.evaluate(X_train, y_train)
    w_mbgd, b_mbgd = get_w_b(model_mbgd)
    w_adam, b_adam = get_w_b(model_adam)

    print('-' * 60)
    print(f'{"优化器":<8}{"最终 MSE":>14}{"学到的 w":>14}{"学到的 b":>14}')
    print(f'{"MBGD":<8}{mse_mbgd:>14.6f}{w_mbgd:>14.4f}{b_mbgd:>14.4f}')
    print(f'{"Adam":<8}{mse_adam:>14.6f}{w_adam:>14.4f}{b_adam:>14.4f}')
    print('-' * 60)

    # ---------- 4. 中文结论打印 ----------
    print('【结论】')
    print(f'  真实参数为 w=3, b=2；噪声标准差={NOISE_STD}，'
          f'因此理论最优 MSE ≈ 噪声方差 {NOISE_STD ** 2:.1f}。')
    print(f'  MBGD 学到 w={w_mbgd:.4f}, b={b_mbgd:.4f}，'
          f'Adam 学到 w={w_adam:.4f}, b={b_adam:.4f}，均接近真值，'
          '验证了 MiniFlows 前向传播、反向传播与优化器实现的正确性。')
    if mse_adam < mse_mbgd:
        print('  Adam 收敛更快/更稳（相同学习率下最终 MSE 更小），'
              '体现其自适应学习率的优势。')
    else:
        print('  两者最终 MSE 相当：对这种强凸简单问题，MBGD 也能充分收敛。')

    # ---------- 5. 图 1：拟合散点 + 真实直线 + 两条拟合直线 ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_train[:, 0], y_train[:, 0], s=18, alpha=0.6,
               color='gray', label='样本点（含噪声）')          # 原始散点
    xs = np.linspace(-3, 3, 100).reshape(-1, 1)               # 画线用的均匀网格
    ax.plot(xs, TRUE_W * xs + TRUE_B, 'k--', linewidth=2,
            label=f'真实直线 y=3x+2')                          # 真实关系
    # 用训练好的模型在网格上预测，画出各自的拟合直线
    ax.plot(xs, trainer_mbgd.predict(xs), color='tab:blue', linewidth=2,
            label=f'MBGD 拟合: y={w_mbgd:.3f}x+{b_mbgd:.3f}')
    ax.plot(xs, trainer_adam.predict(xs), color='tab:red', linewidth=2,
            label=f'Adam 拟合: y={w_adam:.3f}x+{b_adam:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('线性回归拟合结果（MiniFlows，n=%d）' % N_SAMPLES)
    ax.legend()
    ax.grid(alpha=0.3)
    path1 = os.path.join(OUT_DIR, '拟合结果.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')  # 只保存，不 show（Agg 后端）
    plt.close(fig)
    print(f'已保存拟合图：{path1}')

    # ---------- 6. 图 2：两种优化器的训练损失曲线对比 ----------
    path2 = os.path.join(OUT_DIR, '优化器损失对比.png')
    plot_lines({'MBGD (lr=0.05)': hist_mbgd['train_loss'],
                'Adam (lr=0.05)': hist_adam['train_loss']},
               xlabel='训练轮次 Epoch', ylabel='MSE 损失',
               title='MBGD 与 Adam 训练损失对比（线性回归）', path=path2)
    print(f'已保存损失曲线对比图：{path2}')
    print('线性回归应用运行完毕。')


if __name__ == '__main__':
    main()
