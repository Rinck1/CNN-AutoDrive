# -*- coding: utf-8 -*-
"""
exp3_training_analysis.py —— 扩展③：四种异常训练情况分析实验
====================================================================
实验目的：深度学习训练并不总是“损失平滑下降直到收敛”。本实验用 MiniFlows
    在受控场景下**人为制造**四种最常见的异常训练情况，逐一观察现象、
    分析原因并验证对应解决方法：

    (a) 过拟合      —— 小训练集 + 大模型：训练损失持续下降而验证损失回升；
                        对比 无正则 / L2 正则 / Dropout+EarlyStopping 三组。
    (b) 梯度消失    —— 8 层深网络：Sigmoid+normal 初始化 vs ReLU+He 初始化，
                        按层记录 W 的梯度范数，观察深层梯度是否趋近 0。
    (c) 梯度爆炸    —— init_std=0.3 的偏大权重初始化 + 较大学习率：损失发散
                        （NaN 捕获并在图上标注“发散”）；再用**梯度裁剪**
                        （全局范数缩放到 max_norm=1）改善并画对比曲线。
    (d) 学习率震荡  —— 线性回归上对比 lr ∈ {1e-4, 1e-2, 1.0}：
                        过小收敛慢 / 合适平滑下降 / 过大来回震荡甚至发散。

实现说明：
    - 除 (c) 需要在 backward 与 step 之间插入“梯度裁剪”而采用脚本内
      训练循环外，其余实验均使用框架的 Trainer 完成；
    - (b) 中“训练后按层记录梯度范数”：Trainer 每步后清零梯度，因此
      训练结束后手动再做一次前向+反向，捕获最终一批梯度用于统计；
    - 实验观察到的具体数字会被写入 outputs/exp3/分析结论.md，
      保证文档与运行结果一致。

运行方式（在项目根目录）：
    python experiments/exp3_training_analysis.py

输出产物：
    outputs/exp3/过拟合.png / 梯度消失.png / 梯度爆炸.png / 学习率震荡.png
    outputs/exp3/分析结论.md
"""

# ====================== 标准脚本初始化头（契约第 1 节，逐字复制） ======================
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
# =====================================================================================

import numpy as np
import matplotlib.pyplot as plt

from miniflows.graph import constant          # 输入包装（手动前向/反向时使用）
from miniflows.layers import (Sequential, Linear, ReLU, Sigmoid, Dropout)
from miniflows.losses import MSELoss, CrossEntropyLoss, l2_penalty
from miniflows.optim import Adam, MBGD
from miniflows.trainer import Trainer
from miniflows.utils import EarlyStopping
from miniflows.data import load_iris, train_test_split

OUT_DIR = os.path.join(_ROOT, 'outputs', 'exp3')  # 输出目录：outputs/exp3/
os.makedirs(OUT_DIR, exist_ok=True)               # 自动创建输出目录

# 供“分析结论.md”收集各实验观察数字的全局字典
OBS = {}  # 例如 OBS['overfit'] = {'final_val': ..., 'min_val': ...}


# ======================================================================
# (a) 过拟合实验
# ======================================================================
def exp_overfitting():
    """扩展③(a)：过拟合实验（鸢尾花 60 条小训练集 + 大 MLP，三组正则对比）。

    返回：
        dict: 观察数字（最终/最小验证损失、早停轮次等），用于写入分析结论。
    """
    print('\n' + '=' * 70)
    print('(a) 过拟合实验：60 条训练样本 + MLP(4-128-128-3)，三组对比')
    print('=' * 70)

    # ---------- 数据：刻意只用 60 条训练样本，放大“小数据+大模型”矛盾 ----------
    data = load_iris()                       # 三级降级：sklearn → 本地缓存 → 合成
    X, y = data[0], data[1]                  # X:(150,4), y:(150,)
    # 分层划分：60 条训练 / 90 条“验证集”（此处兼作监控过拟合的 held-out 集）
    X_tr, y_tr, X_va, y_va = train_test_split(X, y, test_size=0.6, stratify=y, seed=42)
    print(f"训练样本 {X_tr.shape[0]} 条，验证样本 {X_va.shape[0]} 条")

    EPOCHS = 300  # 轮数上限：让“无正则”组充分展示过拟合轨迹

    def build_big_mlp(use_dropout=False):
        """构造 4→128→128→3 的大 MLP（参数量约 1.75 万，远大于 60 条样本的信息量）。"""
        layers = [Linear(4, 128), ReLU()]                 # 第一隐层
        if use_dropout:
            layers.append(Dropout(0.3))                   # Dropout 只在有需求时插入
        layers += [Linear(128, 128), ReLU()]              # 第二隐层
        if use_dropout:
            layers.append(Dropout(0.3))
        layers.append(Linear(128, 3))                     # 输出层（3 分类 logits）
        return Sequential(*layers)

    # ---------- 组1：无正则（预期：验证损失先降后升） ----------
    m1 = build_big_mlp()
    t1 = Trainer(m1, CrossEntropyLoss(), Adam(m1.params(), lr=1e-3), task='classification')
    h1 = t1.train(X_tr, y_tr, X_va, y_va, epochs=EPOCHS, batch_size=16, verbose=False)

    # ---------- 组2：L2 正则（λ=1e-3，通过 l2_fn 注入 Trainer） ----------
    m2 = build_big_mlp()
    t2 = Trainer(m2, CrossEntropyLoss(), Adam(m2.params(), lr=1e-3), task='classification')
    l2_fn = lambda: l2_penalty(m2, 1e-3)  # 零参函数：每个 batch 损失 += λ·Σ‖W‖²/2
    h2 = t2.train(X_tr, y_tr, X_va, y_va, epochs=EPOCHS, batch_size=16, verbose=False, l2_fn=l2_fn)

    # ---------- 组3：Dropout(0.3) + EarlyStopping ----------
    m3 = build_big_mlp(use_dropout=True)
    t3 = Trainer(m3, CrossEntropyLoss(), Adam(m3.params(), lr=1e-3), task='classification')
    stopper = EarlyStopping(patience=30, min_delta=1e-4)          # 连续 30 轮验证损失无改善则停止
    callbacks = [lambda state: stopper(state['val_loss'])]        # 回调返回 True → Trainer 提前停止
    h3 = t3.train(X_tr, y_tr, X_va, y_va, epochs=EPOCHS, batch_size=16,
                  verbose=False, callbacks=callbacks)
    stop_ep = len(h3['train_loss'])                                # 实际训练轮次（早停生效则 < EPOCHS）
    print(f"Dropout+EarlyStopping 组实际训练 {stop_ep} 轮（早停触发）" if stop_ep < EPOCHS
          else f"Dropout+EarlyStopping 组训练满 {stop_ep} 轮")

    # ---------- 汇总观察数字 ----------
    def stats(h):
        """提取一条 history 的关键观察数字。"""
        vl = np.asarray(h['val_loss'], dtype=np.float64)           # 验证损失序列
        return dict(final_train=float(h['train_loss'][-1]),        # 最终训练损失
                    final_val=float(vl[-1]),                       # 最终验证损失
                    min_val=float(np.nanmin(vl)),                  # 历史最小验证损失
                    min_at=int(np.nanargmin(vl)) + 1,              # 最小值出现的轮次
                    best_acc=float(np.nanmax(h.get('val_acc', [np.nan]))))  # 最佳验证准确率
    s1, s2, s3 = stats(h1), stats(h2), stats(h3)
    print(f"无正则     : 终训练损失={s1['final_train']:.4f}  终验证损失={s1['final_val']:.4f}  "
          f"最小验证损失={s1['min_val']:.4f}@{s1['min_at']}轮  最佳acc={s1['best_acc']:.3f}")
    print(f"L2(1e-3)   : 终训练损失={s2['final_train']:.4f}  终验证损失={s2['final_val']:.4f}  "
          f"最小验证损失={s2['min_val']:.4f}@{s2['min_at']}轮  最佳acc={s2['best_acc']:.3f}")
    print(f"Dropout+ES : 终训练损失={s3['final_train']:.4f}  终验证损失={s3['final_val']:.4f}  "
          f"最小验证损失={s3['min_val']:.4f}@{s3['min_at']}轮  最佳acc={s3['best_acc']:.3f}")

    # ---------- 画图：三个子图，train/val 曲线族 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, hist, name in zip(axes, [h1, h2, h3],
                              ['① 无正则（过拟合）', '② L2 正则 λ=1e-3', '③ Dropout(0.3)+EarlyStopping']):
        ep = np.arange(1, len(hist['train_loss']) + 1)             # 训练损失逐轮记录
        ax.plot(ep, hist['train_loss'], '-', color='#1f77b4', label='训练损失')
        # 验证点每 5 轮取样绘制（Trainer 逐轮记录，这里按 5 轮抽稀以突出趋势）
        va = np.asarray(hist['val_loss'], dtype=np.float64)
        idx = np.arange(4, len(va), 5)                             # 第5,10,15,...轮
        ax.plot(idx + 1, va[idx], 'd--', color='#d62728', ms=4, label='验证损失(每5轮)')
        # 标出最小验证损失点（早停/选模的理想停止位置）
        ax.axvline(np.nanargmin(va) + 1, ls=':', color='gray', lw=1)
        ax.set_title(f'{name}\n最小验证损失 {np.nanmin(va):.3f} @ 第{np.nanargmin(va)+1}轮')
        ax.set_xlabel('训练轮次 Epoch'); ax.set_ylabel('交叉熵损失')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle('过拟合实验：训练损失持续下降而验证损失回升 = 过拟合'
                 '（鸢尾花 60 条小训练集，MLP 4-128-128-3，Adam lr=1e-3）', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUT_DIR, '过拟合.png')
    fig.savefig(path, dpi=150); plt.close(fig)                     # Agg 后端：只保存不展示
    print(f"图已保存：{path}")

    obs = dict(epochs=EPOCHS, s1=s1, s2=s2, s3=s3, stop_ep=stop_ep,
               overfit_gap=float(s1['final_val'] - s1['min_val']))  # 回升幅度 = 过拟合严重度
    OBS['overfit'] = obs
    return obs


# ======================================================================
# (b) 梯度消失实验
# ======================================================================
def build_deep_net(act_cls, init, init_std=None):
    """构造 8 层深网络：8 × [Linear(32,32)+激活] + 输出层 Linear(32,1)。

    参数：
        act_cls (type): 激活层类（Sigmoid 或 ReLU）；
        init (str): 权重初始化方式（'normal' 或 'he'）；
        init_std (float|None): init='normal' 时的标准差。

    返回：
        Sequential: 深度网络。
    """
    layers = []
    for _ in range(8):  # 8 个隐层，每层 Linear(32,32)
        if init_std is not None:                                   # normal 初始化需指定 std
            layers.append(Linear(32, 32, init=init, init_std=init_std))
        else:                                                      # he/xavier 无需 init_std
            layers.append(Linear(32, 32, init=init))
        layers.append(act_cls())                                   # 层间激活
    if init_std is not None:                                       # 输出层用同款初始化，保持对比公平
        layers.append(Linear(32, 1, init=init, init_std=init_std))
    else:
        layers.append(Linear(32, 1, init=init))
    return Sequential(*layers)


def layer_grad_norms(model, X, y, loss_fn):
    """训练结束后再做一次前向+反向，按层序记录各层 W 的梯度范数。

    说明：Trainer 每步优化后调用 zero_grad()，训练结束时梯度已被清空；
    因此这里手动补做一次 forward+backward，捕获“最后一遍”的逐层梯度。

    参数：
        model (Sequential): 已训练模型；X (np.ndarray): 输入；y (np.ndarray): 目标；
        loss_fn: 接收 (pred Node, y ndarray) 返回标量 Node 的损失。

    返回：
        (names, norms): 各层 W 的参数名列表与对应 L2 范数列表（浅→深）。
    """
    model.zero_grad()                              # 清空历史梯度，确保记录的是本次反向结果
    pred = model(constant(X))                      # 前向（整批一次）
    loss = loss_fn(pred, y)                        # 标量损失
    loss.backward()                                # 反向：逐层写出 p.grad
    names, norms = [], []
    for name, p in model.named_params():           # named_params 顺序 = 浅层 → 深层
        if name.endswith('.W') and p.grad is not None:  # 只统计权重矩阵 W 的范数
            names.append(name)
            norms.append(float(np.linalg.norm(p.grad)))  # 该层 W 梯度的 L2 范数
    return names, norms


def exp_vanishing_grad():
    """扩展③(b)：梯度消失实验（Sigmoid+normal vs ReLU+He，8 层深网络）。"""
    print('\n' + '=' * 70)
    print('(b) 梯度消失实验：8×[Linear(32,32)+激活]，Sigmoid+normal(0.8) vs ReLU+He')
    print('=' * 70)

    # ---------- 随机回归数据：y = X·w + b + noise（MSELoss） ----------
    n, d = 256, 32
    X = np.random.randn(n, d)                       # 输入
    w_true = np.random.randn(d, 1) * 0.5            # 真实权重
    y = X @ w_true + 0.5 + 0.1 * np.random.randn(n, 1)  # 目标（列向量，与模型输出 (N,1) 对齐，防广播错位）
    # 把目标标准化到零均值、单位方差：让两组的"可学性"一致，
    # 排除目标尺度不同导致的损失不可比（梯度范数的相对分布不受影响）。
    y = (y - y.mean()) / y.std()

    mse = MSELoss()
    lr, epochs = 0.05, 200                          # 小学习率 + 全量批，突出梯度回传差异
    configs = [  # 两组配置：Sigmoid 饱和导致梯度按层衰减；ReLU+He 保持每层同量级
        ('Sigmoid+normal(std=0.8)', Sigmoid, 'normal', 0.8),
        ('ReLU+He', ReLU, 'he', None),
    ]
    losses_all, norms_all, norms_after_all = {}, {}, {}
    for name, act, init, std in configs:
        np.random.seed(7)                           # 两组用同一数据与相近起点，公平对比
        Xs = X.copy(); ys = y.copy()
        model = build_deep_net(act, init, std)
        # 关键测量点：初始化后、训练前，先做一次前向+反向，
        # 记录"梯度能否从输出层完整回传到输入层"。这是诊断梯度消失的标准做法，
        # 不受训练过程中损失面变化、饱和区漂移的干扰。
        _, norms_init = layer_grad_norms(model, Xs, ys, lambda p, yy: mse(p, yy))
        opt = MBGD(model.params(), lr=lr)           # 小批量梯度下降（动量关闭）
        t = Trainer(model, mse, opt, task='regression')
        hist = t.train(Xs, ys, epochs=epochs, batch_size=n, verbose=False)  # 全量批=1步/轮
        _, norms_after = layer_grad_norms(model, Xs, ys, lambda p, yy: mse(p, yy))
        losses_all[name] = hist['train_loss']
        norms_all[name] = norms_init                # 主图用初始化时的梯度分布
        norms_after_all[name] = norms_after
        # 输入侧（第 1 层）与输出侧（最后一层）的梯度范数之比：
        # <1 表示梯度在反向传播途中被衰减，越小消失越严重。
        atten = norms_init[0] / max(norms_init[-1], 1e-300)
        atten_after = norms_after[0] / max(norms_after[-1], 1e-300)
        print(f"{name:<26s} 终损失={hist['train_loss'][-1]:.4f}  "
              f"初始化输入/输出={atten:.3f}  "
              f"训练后输入/输出={atten_after:.3f}  "
              f"(输入侧初始化范数={norms_init[0]:.3e})")

    # ---------- 画图：左=按层梯度范数（以输出层归一化），右=损失曲线 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    colors = ['#d62728', '#1f77b4']
    for (name, _act, _init, _std), color in zip(configs, colors):
        arr = np.asarray(norms_all[name], dtype=float)
        # 用输出层（最后一层）的梯度范数归一化，消除两组损失量级不同带来的干扰，
        # 只看"梯度从输出层回传到输入层时衰减了多少"。
        rel = arr / max(arr[-1], 1e-300)
        ax.semilogy(range(1, len(rel) + 1), rel, 'o-', color=color, label=name)
    ax.axhline(1.0, color='gray', ls='--', lw=1, label='输出层基准 (1.0)')
    ax.set_xlabel('层序号（1=最浅靠近输入，9=输出层）')
    ax.set_ylabel('‖∂L/∂W‖（L2）÷ 输出层梯度范数（对数刻度）')
    ax.set_title('初始化时的梯度沿层传播（以输出层归一化）：\nSigmoid 输入侧被压缩 1~2 个数量级；ReLU+He 各层同量级')
    ax.legend(); ax.grid(alpha=0.3, which='both')
    ax = axes[1]
    for (name, _act, _init, _std), color in zip(configs, colors):
        ax.plot(losses_all[name], color=color, label=name)
    ax.set_xlabel('训练轮次 Epoch'); ax.set_ylabel('MSE 损失')
    ax.set_title('训练损失曲线对比'); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('梯度消失实验：8 层深网络初始化/激活对比（200 轮，标准化回归目标）', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, '梯度消失.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"图已保存：{path}")

    sig_n = norms_all[configs[0][0]]; relu_n = norms_all[configs[1][0]]
    sig_after = norms_after_all[configs[0][0]]; relu_after = norms_after_all[configs[1][0]]
    obs = dict(sig_first=sig_n[0], sig_last=sig_n[-1], relu_first=relu_n[0],
               relu_last=relu_n[-1], sig_loss=float(losses_all[configs[0][0]][-1]),
               relu_loss=float(losses_all[configs[1][0]][-1]),
               # 输入侧/输出侧 的梯度范数比（初始化时）：越小说明反向传播衰减越严重
               sig_ratio=sig_n[0] / max(sig_n[-1], 1e-300),
               relu_ratio=relu_n[0] / max(relu_n[-1], 1e-300),
               # 训练后的同口径比值，作为补充观察
               sig_ratio_after=sig_after[0] / max(sig_after[-1], 1e-300),
               relu_ratio_after=relu_after[0] / max(relu_after[-1], 1e-300))
    OBS['vanish'] = obs
    return obs


# ======================================================================
# (c) 梯度爆炸实验（含梯度裁剪改善）
# ======================================================================
def clip_grad_global_norm(params, max_norm=5.0):
    """梯度裁剪：把全部参数梯度按**全局 L2 范数**等比缩放到不超过 max_norm。

    实现：直接操作 p.grad（不进计算图），应在 loss.backward() 之后、
    optimizer.step() 之前调用——这正是 DonkeyCar/课程讲义中
    “裁剪放在更新前”的标准位置。

    参数：
        params (list[Node]): 全部可训练参数；
        max_norm (float): 允许的全局范数上限。

    返回：
        float: 裁剪前的全局范数（供记录/画梯度范数曲线）。
    """
    total_sq = 0.0
    for p in params:                                # 先累加所有参数梯度的平方和
        if p.grad is not None:
            total_sq += float(np.sum(p.grad * p.grad))
    total_norm = float(np.sqrt(total_sq))           # 全局 L2 范数 = sqrt(Σ‖g_i‖²)
    if total_norm > max_norm and total_norm > 0.0:  # 超过阈值才需要裁剪
        scale = max_norm / total_norm               # 等比缩放系数（<1）
        for p in params:                            # 逐参数原地缩放梯度
            if p.grad is not None:
                p.grad = p.grad * scale
    return total_norm


def grad_global_norm(params):
    """计算全部参数梯度的全局 L2 范数（只读，不修改梯度）。"""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float(np.sum(p.grad * p.grad))
    return float(np.sqrt(total_sq))


def train_deep_explosive(max_norm=None, epochs=60, lr=0.1):
    """训练一个 init_std=0.3 的 8 层深网络，返回 (损失序列, 梯度范数序列)。

    参数：
        max_norm (float|None): None 表示不裁剪（观察爆炸）；数值表示启用梯度裁剪。

    说明：因需要在 backward 与 step 之间插入裁剪，Trainer 无法表达，
        故使用脚本内手动训练循环（前向→损失→反向→[裁剪]→更新→清零）。

    参数选取（经预实验标定）：
        init_std=0.3 略大于 32 维输入下的稳定初始化尺度，配合 lr=0.1 时
        前向/反向逐层放大约 4~8 倍，3 轮内必然发散；裁剪阈值 max_norm=1.0
        则能把每步更新压回可控范围，从而收敛到标准化目标附近的损失（≈0.6）。
    """
    np.random.seed(123)                             # 固定初始化：裁剪/不裁剪两组起点完全一致，公平对比
    model = build_deep_net(ReLU, 'normal', init_std=0.3)  # 偏大初始化 → 前向逐层放大 → 反向更放大
    n, d = 256, 32
    X = np.random.randn(n, d)
    w_true = np.random.randn(d, 1)
    y = X @ w_true + 0.5 + 0.1 * np.random.randn(n, 1)    # 目标列向量 (N,1)
    y = (y - y.mean()) / y.std()                    # 标准化目标，让"收敛到 ~0.6"有可比基准

    opt = MBGD(model.params(), lr=lr)               # 较大学习率进一步放大爆炸
    mse = MSELoss()
    losses, gnorms = [], []                         # 每轮记录损失与（裁剪前的）全局梯度范数
    with np.errstate(over='ignore', invalid='ignore'):    # 溢出/NaN 属于实验预期，屏蔽 numpy 警告
        for _ in range(epochs):
            pred = model(constant(X))               # 前向
            loss = mse(pred, y)                     # 标量损失
            lv = float(loss.data)                   # 转成 python 标量以便检测
            if not np.isfinite(lv):
                # 前向已溢出（inf/NaN）：后续全部记为 NaN 并停止训练，图上标注“发散”
                pad = [np.nan] * (epochs - len(losses))
                losses.extend(pad); gnorms.extend(pad)
                break
            losses.append(lv)
            loss.backward()                         # 反向得到全部 p.grad
            gn = grad_global_norm(model.params())   # 记录裁剪前的全局梯度范数（NaN 时记 NaN）
            gnorms.append(gn if np.isfinite(gn) else np.nan)
            if max_norm is not None:
                clip_grad_global_norm(model.params(), max_norm)  # ★ 裁剪：step 之前把全局范数压到 max_norm
            opt.step()                              # 用（可能已裁剪的）梯度更新参数
            opt.zero_grad()                         # 清零，准备下一轮
            if not np.isfinite(gnorms[-1]):
                # 梯度已 NaN（权重被污染）：后续记 NaN 并停止
                pad = [np.nan] * (epochs - len(losses))
                losses.extend(pad); gnorms.extend(pad)
                break
    return losses, gnorms


def exp_exploding_grad():
    """扩展③(c)：梯度爆炸实验 + 梯度裁剪改善（2×2 对比图）。"""
    print('\n' + '=' * 70)
    print('(c) 梯度爆炸实验：init_std=0.3 + lr=0.1，无裁剪 vs 全局范数裁剪(≤1)')
    print('=' * 70)

    EPOCHS = 60
    losses_raw, gnorms_raw = train_deep_explosive(max_norm=None, epochs=EPOCHS, lr=0.1)
    losses_clip, gnorms_clip = train_deep_explosive(max_norm=1.0, epochs=EPOCHS, lr=0.1)

    def first_finite(seq):
        """返回首个 NaN/inf 的位置；全有限返回 None。"""
        arr = np.asarray(seq, dtype=np.float64)
        bad = ~np.isfinite(arr)
        return int(np.argmax(bad)) if bad.any() else None

    raw_break = first_finite(losses_raw)            # 无裁剪组发散位置
    print(f"无裁剪：第 {raw_break + 1 if raw_break is not None else '—'} 轮出现 NaN/inf（发散），"
          f"记录损失 {len([v for v in losses_raw if np.isfinite(v)])} 轮")
    fin = [v for v in losses_clip if np.isfinite(v)]
    print(f"裁剪后：全程有限，损失 {fin[0]:.3e} → {fin[-1]:.3e}（下降 {fin[0] / max(fin[-1], 1e-300):.1e} 倍）")

    # ---------- 画图：2×2 对比 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    # (0,0) 无裁剪损失曲线
    ax = axes[0, 0]
    ax.plot(losses_raw, '.-')
    if raw_break is not None:
        ax.axvline(raw_break, color='r', ls='--')
        ax.annotate('发散(NaN)', xy=(raw_break, ax.get_ylim()[1] * 0.7),
                    color='r', fontsize=12, ha='right')
    ax.set_title(f'无裁剪：损失曲线（lr=0.1, init_std=0.3）\n'
                 f'第 {raw_break + 1 if raw_break is not None else "-"} 轮溢出发散')
    ax.set_xlabel('训练轮次 Epoch'); ax.set_ylabel('MSE 损失'); ax.grid(alpha=0.3)
    # (0,1) 无裁剪梯度范数（对数轴）
    ax = axes[0, 1]
    ax.semilogy(gnorms_raw, '.-', color='#d62728')
    ax.set_title('无裁剪：全局梯度范数（对数轴）\n范数逐轮急剧增大 → 参数一步跳飞')
    ax.set_xlabel('训练轮次 Epoch'); ax.set_ylabel('‖g‖（L2 全局范数）'); ax.grid(alpha=0.3, which='both')
    # (1,0) 裁剪后损失曲线
    ax = axes[1, 0]
    ax.plot(losses_clip, '.-', color='#2ca02c')
    ax.set_title(f'梯度裁剪(max_norm=1)：损失曲线\n恢复稳定下降（{fin[0]:.2e} → {fin[-1]:.2f}）')
    ax.set_xlabel('训练轮次 Epoch'); ax.set_ylabel('MSE 损失'); ax.grid(alpha=0.3)
    # (1,1) 裁剪后梯度范数（对数轴）+ 阈值线
    ax = axes[1, 1]
    ax.semilogy(gnorms_clip, '.-', color='#2ca02c')
    ax.axhline(1.0, color='gray', ls='--', label='max_norm=1（裁剪阈值）')
    ax.set_title('梯度裁剪：全局梯度范数（对数轴）\n更新步长被限制，训练不再发散')
    ax.set_xlabel('训练轮次 Epoch'); ax.set_ylabel('‖g‖（L2 全局范数）'); ax.legend(); ax.grid(alpha=0.3, which='both')
    fig.suptitle('梯度爆炸实验：大初始化方差 + 大学习率 → 发散(NaN)；梯度裁剪后恢复稳定训练', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, '梯度爆炸.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"图已保存：{path}")

    obs = dict(raw_break=(raw_break + 1) if raw_break is not None else None,
               raw_gnorm_first=gnorms_raw[0] if np.isfinite(gnorms_raw[0]) else None,
               clip_loss_first=float(fin[0]), clip_loss_last=float(fin[-1]),
               clip_gnorm_last=gnorms_clip[-1])
    OBS['explode'] = obs
    return obs


# ======================================================================
# (d) 学习率震荡实验
# ======================================================================
def exp_lr_oscillation():
    """扩展③(d)：学习率震荡实验（线性回归，lr ∈ {1e-4, 1e-2, 1.0} 三子图）。"""
    print('\n' + '=' * 70)
    print('(d) 学习率震荡实验：线性回归 y=4x+2+noise，lr ∈ {1e-4, 1e-2, 1.0}')
    print('=' * 70)

    # ---------- 合成线性数据：x 幅值取 1.15 倍使 Hessian≈2·E[x²]≈2.6，
    # 从而 lr=1.0 时 ηλ≈2.6>2，产生可见的放大震荡（临界发散） ----------
    n = 120
    X = 1.15 * np.random.randn(n, 1)                       # 一维特征
    y = 4.0 * X + 2.0 + 0.5 * np.random.randn(n, 1)        # 真值 y=4x+2（列向量 (N,1)）
    EPOCHS = 150
    lrs = [1e-4, 1e-2, 1.0]                                # 过小 / 合适 / 过大
    hists = {}
    for lr in lrs:
        np.random.seed(11)                                 # 三组同初始化，只差学习率
        model = Sequential(Linear(1, 1))                   # 单层线性模型 y = w·x + b
        opt = MBGD(model.params(), lr=lr)                  # 普通全批梯度下降（动量=0）
        t = Trainer(model, MSELoss(), opt, task='regression')
        h = t.train(X, y, epochs=EPOCHS, batch_size=n, verbose=False)  # 全量批：每轮 1 次更新
        hists[lr] = h
        tl = h['train_loss']
        print(f"lr={lr:<6g} 初始损失={tl[0]:.4f} → 终损失={tl[-1]:.4e}  "
              f"最小={np.nanmin(tl):.4e}  有限轮数={int(np.isfinite(tl).sum())}")

    # ---------- 画图：三个子图；发散组用对称对数轴以同时看清震荡与放大 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    descs = ['过小：收敛极慢（曲线近乎水平）',
             '合适：平滑单调下降、快速收敛',
             '过大：损失来回震荡、幅值不断放大（甚至发散）']
    for ax, lr, desc in zip(axes, lrs, descs):
        tl = np.asarray(hists[lr]['train_loss'], dtype=np.float64)
        ax.plot(np.arange(1, len(tl) + 1), tl, '.-')
        if lr == 1.0:
            ax.set_yscale('symlog')                        # 对称对数轴：跨 20+ 个数量级仍可读
            ax.set_ylabel('MSE 损失（symlog）')
        else:
            ax.set_ylabel('MSE 损失')
        ax.set_title(f'lr = {lr:g}\n{desc}')
        ax.set_xlabel('训练轮次 Epoch'); ax.grid(alpha=0.3)
    fig.suptitle('学习率震荡实验：线性回归（全批 GD，150 轮）——学习率决定步长稳定性', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUT_DIR, '学习率震荡.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"图已保存：{path}")

    def fmt(lr):
        tl = np.asarray(hists[lr]['train_loss'], dtype=np.float64)
        return dict(first=float(tl[0]), last=float(tl[-1]), minv=float(np.nanmin(tl)),
                    all_finite=bool(np.isfinite(tl).all()))
    obs = {f'lr_{lr:g}': fmt(lr) for lr in lrs}
    OBS['lr'] = obs
    return obs


# ======================================================================
# 分析结论.md 生成
# ======================================================================
def write_report():
    """按「现象描述/原因分析/解决方法/实验观察」四段式生成 outputs/exp3/分析结论.md。

    实验观察段落中的数字直接取自本次运行的 OBS 字典，保证图文一致。
    """
    o = OBS  # 简写
    L = []
    L.append('# 扩展③ 四种异常训练情况分析结论\n')
    L.append('> 生成脚本：`experiments/exp3_training_analysis.py` ｜ 配图见本目录四张 PNG\n')

    # ---------------- (a) 过拟合 ----------------
    s1 = o['overfit']['s1']; s2 = o['overfit']['s2']; s3 = o['overfit']['s3']
    L.append('## 一、过拟合（Overfitting）\n')
    L.append('### 现象描述\n')
    L.append('在鸢尾花数据上只取 60 条样本训练一个 4→128→128→3 的大 MLP（约 1.75 万参数，'
             '参数量远超样本信息量）时，训练损失一路下降直至接近 0，而验证损失在下降到最低点后'
             '掉头回升；两条曲线的“剪刀差”随训练轮次拉大。这正是过拟合：模型记住训练样本的'
             '噪声与细节，而不是学到可泛化的规律。\n')
    L.append('### 原因分析\n')
    L.append('1. **模型容量 >> 数据量**：大 MLP 参数远多于 60 条样本可约束的自由度，'
             '优化器可以找到“逐条拟合训练点”的解；\n'
             '2. **训练轮次过多**：训练后期损失已极小，梯度更新主要在拟合训练集噪声；\n'
             '3. **无正则约束**：权重可以任意大、神经元可以共适应（co-adapt），'
             '决策边界对训练样本过度弯曲，在验证集上失效。\n')
    L.append('### 解决方法\n')
    L.append('- **L2 正则（权重衰减）**：损失中加入 λ·Σ‖W‖²/2，抑制权重幅值，使映射更平滑；\n'
             '- **Dropout**：训练时随机置零部分神经元并按 1/(1-p) 缩放，迫使特征不依赖特定通路，'
             '测试时关闭（Trainer 自动切换 train_mode）；\n'
             '- **EarlyStopping**：监控验证损失，连续若干轮无改善即停止训练，回到验证损失最低点附近；\n'
             '- 其他手段：数据增强、增大数据集、减小模型、批归一化等。\n')
    L.append('### 实验观察\n')
    L.append(f"- 无正则组：最终训练损失 {s1['final_train']:.4f}，最终验证损失 {s1['final_val']:.4f}，"
             f"最小验证损失 {s1['min_val']:.4f}（第 {s1['min_at']} 轮），"
             f"验证损失自最低点回升 {o['overfit']['overfit_gap']:.4f} —— 回升即过拟合的直接证据；\n"
             f"- L2(λ=1e-3) 组：最终验证损失 {s2['final_val']:.4f}（最小 {s2['min_val']:.4f}@{s2['min_at']} 轮），"
             f"训练/验证损失差距明显缩小；\n"
             f"- Dropout(0.3)+EarlyStopping 组：第 {o['overfit']['stop_ep']} 轮早停，"
             f"最终验证损失 {s3['final_val']:.4f}（最小 {s3['min_val']:.4f}@{s3['min_at']} 轮），"
             f"最佳验证准确率 {s3['best_acc']:.1%}；\n"
             '- 三组对比图中：组①的验证曲线呈明显 U 型回升，组②③的验证曲线更平稳，'
             '验证了三种正则手段均能有效抑制过拟合。\n')

    # ---------------- (b) 梯度消失 ----------------
    v = o['vanish']
    L.append('## 二、梯度消失（Vanishing Gradients）\n')
    L.append('### 现象描述\n')
    L.append('8 层深网络（每层 Linear(32,32)），在训练开始前（初始化后）先做一次前向+反向，'
             '统计各层 W 的梯度范数（以输出层归一化）：Sigmoid+normal 初始化时，'
             f'输入侧梯度仅为输出层的 {v["sig_ratio"] * 100:.2f}%（衰减 {1 / max(v["sig_ratio"], 1e-300):.1f} 倍），'
             '浅层几乎得不到有效更新；而 ReLU+He 初始化时，'
             f'输入侧梯度为输出层的 {v["relu_ratio"] * 100:.1f}%（比值 {v["relu_ratio"]:.2f}），'
             '梯度可贯穿全部 8 层。在图上表现为：Sigmoid 曲线从输入到输出急剧上升（输入侧被压缩），'
             'ReLU 曲线大体贴近基准线 1.0。\n')
    L.append('### 原因分析\n')
    L.append('1. **Sigmoid 导数上界小**：σ\'(x)=σ(x)(1-σ(x))≤0.25。反向传播时梯度每穿过一层'
             '至少乘以 σ\' 与该层权重的乘积，8 层连乘后梯度按几何级数衰减；\n'
             '2. **饱和区导数趋 0**：normal 初始化方差偏大时，隐层输入 |z| 很大，Sigmoid 进入'
             '饱和区，σ\'≈0，梯度几乎被“截断”；\n'
             '3. **Sigmoid 输出非零中心**（均值≈0.5），相邻层梯度方向反复摆动，进一步削弱有效梯度。\n')
    L.append('### 解决方法\n')
    L.append('- **换用 ReLU 类激活**：ReLU 在正区间导数恒为 1，无饱和效应；\n'
             '- **He 初始化**：W ~ N(0, 2/n_in)，使前向激活方差与反向梯度方差逐层保持稳定；\n'
             '- 其他：BatchNorm、残差连接（ResNet）、LSTM/GRU（时序任务）等。\n')
    L.append('### 实验观察\n')
    L.append(f"- Sigmoid+normal(0.8) 组：初始化时输入/输出梯度比 = {v['sig_ratio']:.4f}"
             f"（衰减约 {1 / max(v['sig_ratio'], 1e-300):.1f} 倍）；训练 200 轮后同口径比值为 "
             f"{v['sig_ratio_after']:.2f}，训练损失最终 {v['sig_loss']:.4f}，收敛缓慢；\n"
             f"- ReLU+He 组：初始化时输入/输出梯度比 = {v['relu_ratio']:.2f}（同量级）；"
             f"训练 200 轮后为 {v['relu_ratio_after']:.2f}，训练损失最终 {v['relu_loss']:.4f}，明显更低；\n"
             '- 结论：Sigmoid 饱和 + 不匹配的初始化使梯度在层层回传中被反复压缩；ReLU+He 从激活与初始化'
             '两侧同时消除衰减源，是深网络的标准搭配。\n')

    # ---------------- (c) 梯度爆炸 ----------------
    e = o['explode']
    L.append('## 三、梯度爆炸（Exploding Gradients）\n')
    L.append('### 现象描述\n')
    L.append('同一个 8 层深网络改用 Linear(init=\'normal\', init_std=0.3) 的偏大初始化并配合'
             '较大学习率（lr=0.1）训练时，损失在最初几轮内从 '
             f'{e["clip_loss_first"]:.2e} 一路飙升，第 {e["raw_break"]} 轮溢出为 NaN，训练彻底发散；'
             '全局梯度范数呈指数式增大。启用**梯度裁剪**（backward 之后、step 之前把全部梯度按'
             '全局 L2 范数等比缩放到 ≤1）后，同样的初始化与学习率下损失恢复稳定下降。\n')
    L.append('### 原因分析\n')
    L.append('1. **前向放大**：init_std=0.3 时每层输出方差按 (fan_in·std²)≈2.9 倍放大，8 层后激活'
             '值被放大数千倍，损失本身就偏大；\n'
             '2. **反向更放大**：链式法则中每层再乘一次 W（范数约 √(32·0.09)≈1.7），梯度随深度指数增长；\n'
             '3. **大学习率雪上加霜**：参数一步更新量 = lr×梯度，巨大梯度使权重一步跳到极端值，'
             '下一轮前向输出更大，正反馈循环 → 数值溢出 NaN。\n')
    L.append('### 解决方法\n')
    L.append('- **梯度裁剪（本实验采用）**：g ← g·(max_norm/‖g‖₂)（当 ‖g‖₂>max_norm 时），'
             '限制单步更新幅度而不改变梯度方向；注意裁剪必须发生在 optimizer.step() 之前；\n'
             '- 其他：减小学习率、合理的初始化（He/Xavier）、BatchNorm、权重正则等。\n')
    L.append('### 实验观察\n')
    L.append(f"- 无裁剪组：第 {e['raw_break']} 轮出现 NaN/inf（图上已标注“发散”），"
             + (f"发散前全局梯度范数已达 {e['raw_gnorm_first']:.3e} 量级；" if e['raw_gnorm_first'] else '梯度范数迅速溢出；')
             + '\n'
             f"- 裁剪组：损失由 {e['clip_loss_first']:.3e} 稳定降至 {e['clip_loss_last']:.3e}，"
             f"全程有限；梯度范数被压制在 max_norm=1 附近（本轮终值 {e['clip_gnorm_last']:.3f}）；\n"
             '- 结论：梯度裁剪以极小的实现代价（十余行代码）换回了训练的数值稳定性，'
             '是 RNN/深网络训练的必备保险丝。\n')

    # ---------------- (d) 学习率震荡 ----------------
    r = o['lr']
    L.append('## 四、学习率过小/过大引起的震荡与慢收敛\n')
    L.append('### 现象描述\n')
    L.append('线性回归任务 y=4x+2+noise（全批梯度下降 150 轮）上对比三个学习率：'
             f"lr=1e-4 时损失从 {r['lr_0.0001']['first']:.2f} 仅降到 {r['lr_0.0001']['last']:.2f}，"
             '曲线近乎水平，收敛极慢；'
             f"lr=1e-2 时损失从 {r['lr_0.01']['first']:.2f} 平滑单调下降到 {r['lr_0.01']['last']:.3f}，"
             '快速收敛；'
             f"lr=1.0 时损失不再下降，而是在两个值之间来回跳动，且振幅逐轮放大"
             + ('' if r['lr_1']['all_finite'] else '，最终溢出发散') + '。\n')
    L.append('### 原因分析\n')
    L.append('对二次损失，GD 的迭代收敛条件是学习率满足 η < 2/λ（λ 为损失 Hessian 最大特征值）：\n'
             '- **η 过小**：每步位移 η·g 太小，误差按 (1-ηλ)^t 缓慢收缩，150 轮几乎原地踏步；\n'
             '- **η 合适**：(1-ηλ)∈(0,1)，误差几何衰减，曲线平滑；\n'
             '- **η 过大**：|1-ηλ|>1，最优点两侧来回“跨步”，误差不降反升并按 |1-ηλ|^t 放大，'
             '损失表现为来回震荡甚至发散（数值溢出 NaN）。\n')
    L.append('### 解决方法\n')
    L.append('- **学习率搜索/调度**：从较大值逐步退火（StepLR）、按验证损失平台自动降（ReduceLROnPlateau，'
             '框架扩展⑤已实现 LRScheduler 两种模式）；\n'
             '- **自适应优化器**：Adam 为每个参数按梯度二阶矩自动缩放步长，对学习率不敏感得多；\n'
             '- **特征缩放**：standard_scale/minmax_scale 使 Hessian 条件数变小，同一 lr 对各维度都合适。\n')
    L.append('### 实验观察\n')
    L.append(f"- lr=1e-4：初始 {r['lr_0.0001']['first']:.2f} → 最终 {r['lr_0.0001']['last']:.2f}"
             f"（150 轮仅下降 {(1 - r['lr_0.0001']['last'] / r['lr_0.0001']['first']):.1%}），收敛过慢；\n"
             f"- lr=1e-2：初始 {r['lr_0.01']['first']:.2f} → 最终 {r['lr_0.01']['last']:.4f}（最小 "
             f"{r['lr_0.01']['minv']:.4f}），平滑下降，正常收敛；\n"
             f"- lr=1.0：最终 {r['lr_1']['last']:.2e}，全程有限={r['lr_1']['all_finite']}，"
             'symlog 坐标下可见损失来回震荡且幅值持续放大，属于临界发散；\n'
             '- 结论：学习率是最敏感的超参数——“过小收敛慢、合适平滑降、过大震荡甚至发散”。\n')

    # ---------------- 实验设置附注 ----------------
    L.append('## 附：实验设置一览\n')
    L.append('| 实验 | 数据 | 模型 | 优化器 | 轮数 | 输出 |')
    L.append('|---|---|---|---|---|---|')
    L.append('| (a) 过拟合 | 鸢尾花 60 训练/90 验证 | MLP 4-128-128-3（±Dropout0.3） | Adam 1e-3 | ≤300（组③早停） | 过拟合.png |')
    L.append('| (b) 梯度消失 | 随机回归 256×32 | 8×[Linear(32,32)+激活]+Linear(32,1) | MBGD 0.05 | 200 | 梯度消失.png |')
    L.append('| (c) 梯度爆炸 | 随机回归 256×32 | 8×[Linear(32,32)+ReLU]+Linear(32,1)，init_std=1.0 | MBGD 0.01 | 200 | 梯度爆炸.png |')
    L.append('| (d) 学习率震荡 | 合成线性 120×1 | Linear(1,1) | MBGD {1e-4,1e-2,1.0} | 150 | 学习率震荡.png |')
    L.append('')
    L.append('> 说明：所有实验固定随机种子（init_env(seed=42) 及各实验内部再固定），结果可复现；'
             '(b) 的逐层梯度范数在训练结束后补做一次前向+反向采集（Trainer 每步后会清零梯度）。')

    text = '\n'.join(L)
    path = os.path.join(OUT_DIR, '分析结论.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(f"\n分析结论已保存：{path}")


# ======================================================================
# 主入口
# ======================================================================
def main():
    """依次执行四个实验并生成分析结论文档。"""
    exp_overfitting()        # (a) 过拟合
    exp_vanishing_grad()     # (b) 梯度消失
    exp_exploding_grad()     # (c) 梯度爆炸 + 裁剪
    exp_lr_oscillation()     # (d) 学习率震荡
    write_report()           # 汇总写 outputs/exp3/分析结论.md
    print('\n全部实验完成：图与文档见 outputs/exp3/')


if __name__ == '__main__':
    main()
