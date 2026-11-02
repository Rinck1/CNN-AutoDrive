# -*- coding: utf-8 -*-
"""
iris.py —— 基于 MiniFlows 的鸢尾花多分类应用：线性 Softmax 回归 vs 两层 MLP
==========================================================================

功能概述（课程基本任务③）：
    1. 用 miniflows.data.load_iris 加载鸢尾花数据（150 条，4 特征，3 类）；
    2. 预处理：train_test_split(stratify=y, test_size=0.3) 后，
       用训练集统计量 standard_scale 标准化训练集，apply_scale 变换测试集
       （严格避免测试集信息泄漏进预处理统计量）；
    3. 对比两个模型（均 CrossEntropyLoss + Adam(lr=1e-3)，100 轮，batch=16）：
       - 模型①：线性 Softmax 回归  Linear(4,3)
       - 模型②：两层 MLP           Linear(4,16) -> ReLU -> Linear(16,3)
    4. 打印两者测试集准确率（MLP 应 ≥90%）；
    5. 保存产物：
       - outputs/iris/训练损失对比.png（两个模型每轮训练损失曲线）
       - outputs/iris/测试准确率对比.png（两个模型每轮验证准确率曲线）
    6. 打印中文结论：分析非线性模型（MLP）相对线性模型的收益。

超参选择理由：
    - standard_scale（均值0/方差1）：鸢尾花 4 个特征量纲差异大（花瓣长 cm 级 vs
      宽 0.1cm 级），标准化后 Adam 步长对各维度一致，收敛更稳；
    - 隐层 16 个神经元：150 条小数据集上容量适中，既能拟合非线性又不至于过拟合；
    - lr=1e-3：Adam 默认量级，小数据集上稳定；batch=16：约为训练集的 1/6，
      每轮 7 次参数更新，100 轮共 700 次更新，充分收敛；
    - init='xavier'：配合 tanh/sigmoid 类收敛理论，这里对线性输出层也适用，
      小网络下 He/Xavier 差异不大，统一用默认初始化即可（注释保留说明）。

运行方式（在项目根目录）：
    python applications/iris.py
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

from miniflows.layers import Linear, Sequential, ReLU      # 层与容器
from miniflows.losses import CrossEntropyLoss              # 多分类交叉熵
from miniflows.optim import Adam                           # Adam 优化器
from miniflows.trainer import Trainer                      # 统一训练器
from miniflows.data import load_iris, standard_scale, apply_scale, train_test_split
from miniflows.visualization import plot_lines             # 多曲线对比图

# ------------------------- 全局常量（超参数集中管理） -------------------------
OUT_DIR = init_env(seed=42, output_dir='outputs/iris')  # 输出目录，自动创建
TEST_SIZE = 0.3     # 测试集占比：45 条，符合题目要求
EPOCHS = 100        # 训练轮数：全量仅 105 条训练样本，100 轮很快且充分
BATCH_SIZE = 16     # 小批量大小：105 条 -> 每轮 7 批，梯度更新带适度噪声
LR = 1e-3           # Adam 学习率：契约第 13 节推荐值，小数据集上稳定
HIDDEN = 16         # MLP 隐层神经元数：契约推荐 4->16->3


def load_and_preprocess():
    """加载鸢尾花数据并完成划分 + 标准化预处理。

    流程（严格防止数据泄漏）：
        1) load_iris 读入 (150,4) 特征与 (150,) 标签；
        2) train_test_split 分层抽样（stratify=y）按 7:3 划分，
           保证训练/测试集中三类比例一致（各约 1/3）；
        3) 只在训练集上计算均值/标准差（standard_scale），
           再用同一统计量 apply_scale 变换测试集。

    返回：
        (X_tr, y_tr, X_te, y_te, class_names): 标准化后的训练/测试数据与类别名。
    """
    X, y, class_names = load_iris()   # X:(150,4) float64, y:(150,) int
    print(f'鸢尾花数据加载完成：X.shape={X.shape}, 类别={class_names}')

    # 分层划分：stratify=y 保证各类样本按比例进入训练/测试集
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              shuffle=True, stratify=y, seed=42)
    print(f'划分完成：训练集 {X_tr.shape[0]} 条，测试集 {X_te.shape[0]} 条（分层抽样）')

    # 标准化：只用训练集统计量（防止测试集信息泄漏），测试集套用同一 scaler
    X_tr, scaler = standard_scale(X_tr)   # scaler={'mean':(4,), 'std':(4,)}
    X_te = apply_scale(X_te, scaler)      # 用训练集的 mean/std 变换测试集
    print('标准化完成（统计量仅来自训练集，测试集用 apply_scale 变换）')
    return X_tr, y_tr, X_te, y_te, class_names


def build_softmax_regression():
    """构建模型①：线性 Softmax 回归。

    结构：Linear(4,3)，输出 3 类 logits；
    CrossEntropyLoss 内部做 log-softmax + NLL，因此无需显式 Softmax 层。

    返回：
        Sequential: 线性多分类模型。
    """
    # 4 特征 -> 3 类；等价于 softmax 回归（决策面为 3 条直线）
    return Sequential(Linear(4, 3))


def build_mlp():
    """构建模型②：两层 MLP（一个隐层）。

    结构：Linear(4,16) -> ReLU -> Linear(16,3)；
    ReLU 引入非线性，使模型能拟合弯曲的类间边界。

    返回：
        Sequential: 两层 MLP 模型。
    """
    # 隐层 16 神经元：小数据集上容量适中；ReLU 激活引入非线性
    return Sequential(Linear(4, HIDDEN), ReLU(), Linear(HIDDEN, 3))


def train_one(model_name, model, X_tr, y_tr, X_te, y_te):
    """用统一的超参训练一个模型并返回其 Trainer 与训练历史。

    参数：
        model_name (str): 模型名称（仅用于打印）。
        model (Sequential): 待训练模型。
        X_tr, y_tr: 训练数据（已标准化）。
        X_te, y_te: 测试数据（作为验证集逐轮记录曲线）。

    返回：
        (trainer, history): 训练器与历史 {'train_loss': [...], 'val_acc': [...]}。
    """
    # CrossEntropyLoss：logits(N,3) + 整型标签(N,) -> 标量损失
    loss_fn = CrossEntropyLoss()
    # Adam(lr=1e-3)：契约推荐；对全部可训练参数统一更新
    optimizer = Adam(model.params(), lr=LR)
    trainer = Trainer(model, loss_fn, optimizer, task='classification')
    print(f'\n开始训练模型 [{model_name}]（epochs={EPOCHS}, batch={BATCH_SIZE}, '
          f'Adam lr={LR}）...')
    # 测试集作为验证集传入 -> history 记录 val_loss / val_acc 逐轮曲线
    # verbose=False：100 轮逐轮打印冗长，结果由主流程汇总打印
    history = trainer.train(X_tr, y_tr, X_val=X_te, y_val=y_te,
                          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=False)
    return trainer, history


# ================================ 主流程 ================================
def main():
    """鸢尾花主流程：预处理 -> 双模型训练对比 -> 打印结论 -> 保存对比图。"""
    print('=' * 60)
    print('MiniFlows 鸢尾花多分类：Softmax 回归 vs 两层 MLP')
    print('=' * 60)

    # ---------- 1. 数据加载与预处理 ----------
    X_tr, y_tr, X_te, y_te, class_names = load_and_preprocess()

    # ---------- 2. 训练两个对比模型 ----------
    model_lin = build_softmax_regression()          # 模型① 线性 Softmax 回归
    trainer_lin, hist_lin = train_one('线性 Softmax 回归 Linear(4,3)',
                                      model_lin, X_tr, y_tr, X_te, y_te)
    model_mlp = build_mlp()                         # 模型② 两层 MLP
    trainer_mlp, hist_mlp = train_one(f'两层 MLP Linear(4,{HIDDEN})+ReLU+Linear({HIDDEN},3)',
                                      model_mlp, X_tr, y_tr, X_te, y_te)

    # ---------- 3. 测试集准确率评估 ----------
    _, acc_lin = trainer_lin.evaluate(X_te, y_te, batch_size=256)   # argmax 命中率
    _, acc_mlp = trainer_mlp.evaluate(X_te, y_te, batch_size=256)
    print('\n' + '-' * 60)
    print(f'{"模型":<28}{"测试准确率":>12}')
    print(f'{"线性 Softmax 回归 Linear(4,3)":<28}{acc_lin * 100:>11.2f}%')
    print(f'{"两层 MLP 4-16-ReLU-3":<28}{acc_mlp * 100:>11.2f}%')
    print('-' * 60)

    # ---------- 4. 中文结论 ----------
    print('【结论】')
    print(f'  线性 Softmax 回归测试准确率 {acc_lin * 100:.2f}%，'
          f'两层 MLP 测试准确率 {acc_mlp * 100:.2f}%。')
    if acc_mlp >= 0.9:
        print(f'  MLP 准确率 ≥90%，满足任务要求。')
    if acc_mlp > acc_lin:
        print('  MLP 引入 ReLU 非线性隐层，能拟合弯曲的类间边界，'
              '在特征空间线性不可分的样本上更占优势；')
        print('  但鸢尾花数据本身近似线性可分，两者差距通常不大（几个百分点内），'
              '说明"模型容量应与问题复杂度匹配"。')
    else:
        print('  线性模型已足够拟合鸢尾花数据（近似线性可分），'
              'MLP 并未带来明显收益，且小数据集上更复杂的模型更易过拟合。')

    # ---------- 5. 图 1：两个模型训练损失对比 ----------
    path1 = os.path.join(OUT_DIR, '训练损失对比.png')
    plot_lines({'线性 Softmax 回归': hist_lin['train_loss'],
                '两层 MLP': hist_mlp['train_loss']},
               xlabel='训练轮次 Epoch', ylabel='交叉熵损失',
               title='鸢尾花：两模型训练损失对比（Adam lr=1e-3, batch=16）',
               path=path1)
    print(f'已保存训练损失对比图：{path1}')

    # ---------- 6. 图 2：两个模型验证（测试）准确率对比 ----------
    path2 = os.path.join(OUT_DIR, '测试准确率对比.png')
    plot_lines({'线性 Softmax 回归': hist_lin['val_acc'],
                '两层 MLP': hist_mlp['val_acc']},
               xlabel='训练轮次 Epoch', ylabel='验证准确率',
               title='鸢尾花：两模型测试准确率对比', path=path2)
    print(f'已保存测试准确率对比图：{path2}')
    print('鸢尾花应用运行完毕。')


if __name__ == '__main__':
    main()
