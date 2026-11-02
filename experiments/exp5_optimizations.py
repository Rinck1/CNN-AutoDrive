# -*- coding: utf-8 -*-
"""
experiments/exp5_optimizations.py —— 扩展⑤：框架优化功能演示
================================================================

围绕 MiniFlows 的训练增强能力做五组对比实验，每组出一张图 + 中文结论：

a) 优化器对比      —— 线性回归合成数据上 MBGD vs MBGD+momentum vs Adam 损失曲线；
b) Dropout 正则    —— 鸢尾花 60 样本训练大 MLP(4->64->3)，有无 Dropout 的 train/val 曲线；
c) 学习率调度      —— LRScheduler 阶梯衰减(step) 与 自适应衰减(plateau) 的 lr 变化曲线；
d) 早停            —— 过拟合训练中被 EarlyStopping 提前停止的轮次演示；
e) 保存/加载一致性 —— 训练 -> save_model -> 重新 load_model -> 预测结果完全一致验证。

运行方式（项目根目录）：python experiments/exp5_optimizations.py
输出目录：outputs/exp5/
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
import matplotlib.pyplot as plt                              # 绘图（Agg 后端已由 init_env 设置）

from miniflows.graph import Node, param                     # 节点类 + 参数节点（调度实验用）
from miniflows.layers import Sequential, Linear, ReLU, Dropout
from miniflows.losses import MSELoss, CrossEntropyLoss
from miniflows.optim import MBGD, Adam
from miniflows.trainer import Trainer
from miniflows.utils import (set_seed, EarlyStopping, LRScheduler,
                             save_model, load_model, count_params)
from miniflows.data import load_iris, train_test_split, standard_scale
from miniflows.visualization import plot_lines, plot_loss_curve

OUT_DIR = os.path.join(_ROOT, "outputs", "exp5")             # 契约第 12 节约定的输出目录
os.makedirs(OUT_DIR, exist_ok=True)                          # 自动创建输出目录
conclusions = []                                             # 收集各小节结论，最后写 md


# ======================================================================
# 小节 a：优化器对比（MBGD / +momentum / Adam）
# ======================================================================
def sec_a_optimizers():
    """在线性回归合成数据上对比三种优化器的收敛速度。"""
    print("\n" + "=" * 60)
    print("小节 a：优化器对比 —— MBGD vs MBGD+momentum vs Adam")
    print("=" * 60)
    # ---------- 1. 生成线性回归合成数据：y = X@w + b + 噪声 ----------
    set_seed(42)                                             # 全局种子保证数据一致
    rs = np.random.RandomState(42)
    X = rs.uniform(-1, 1, size=(120, 3))                     # 120 条 3 维特征
    w_true = np.array([[1.5], [-2.0], [0.7]])                # 真实权重（列向量）
    y = X @ w_true + 0.5 + rs.normal(0, 0.3, size=(120, 1))  # 目标值（注意保持 (N,1) 与预测同形）
    print(f"数据: X{X.shape}, y{y.shape}, 真实 w={w_true.ravel()}, b=0.5")

    # ---------- 2. 三种优化器各自从头训练同一结构 ----------
    curves = {}                                              # 记录三种优化器的损失曲线
    w_learned = {}                                           # 记录学到的权重
    configs = [
        ("MBGD", lambda ps: MBGD(ps, lr=0.05)),              # 普通 MBGD
        ("MBGD+momentum", lambda ps: MBGD(ps, lr=0.05, momentum=0.9)),  # 动量加速
        ("Adam", lambda ps: Adam(ps, lr=0.05)),              # 自适应学习率
    ]
    for name, make_opt in configs:
        set_seed(42)                                         # 每次实验完全同起点（同初始化）
        model = Sequential(Linear(3, 1, name="fc"))          # 单层线性回归模型
        opt = make_opt(model.params())                       # 构造对应优化器
        trainer = Trainer(model, MSELoss(), opt, task="regression", seed=42)
        history = trainer.train(X, y, epochs=80, batch_size=16, verbose=False)
        curves[name] = history["train_loss"]                 # 记录损失曲线
        w_learned[name] = model.named_params()[0][1].data.ravel()  # 学到的权重
        print(f"{name:<14}: 初始损失 {history['train_loss'][0]:.4f} -> "
              f"最终损失 {history['train_loss'][-1]:.4f}，学到的 W={np.round(w_learned[name], 3)}")

    # ---------- 3. 画对比曲线 ----------
    path = os.path.join(OUT_DIR, "优化器对比.png")
    plot_lines(curves, xlabel="轮次 Epoch", ylabel="训练损失 MSE",
               title="优化器对比：MBGD vs MBGD+momentum vs Adam（线性回归）", path=path)
    print(f"[已保存] {path}")
    conclusions.append(
        "a) 优化器对比：三种优化器最终都能把 MSE 降到接近噪声水平，"
        "但 Adam（自适应步长）前期下降最快；MBGD+momentum 借助动量在震荡方向"
        "互相抵消、一致方向加速，收敛明显快于普通 MBGD；普通 MBGD 步长固定、"
        "收敛最平稳也最慢。动量与自适应学习率是深度学习调参的第一杠杆。")
    return curves


# ======================================================================
# 小节 b：Dropout 正则效果（鸢尾花 60 样本 + 大 MLP）
# ======================================================================
def _iris_subset(n_total=60):
    """加载鸢尾花并按类别均衡取 n_total 个样本，再分层划分为训练/验证集。

    返回：
        (X_train, y_train, X_val, y_val, class_names)
    """
    X, y, class_names = load_iris()                          # 全量 150 条
    # 按类别轮流取样（round-robin）得到类别均衡子集：60 = 每类 20
    picked, k = [], 0
    classes = np.unique(y)
    class_idx = {c: np.where(y == c)[0] for c in classes}
    while len(picked) < n_total:
        for c in classes:                                    # 每轮从每个类取 1 个
            if len(picked) >= n_total:
                break
            if k < len(class_idx[c]):
                picked.append(class_idx[c][k])
        k += 1
    picked = np.array(sorted(picked))                        # 保持原相对顺序
    X_sub, y_sub = X[picked], y[picked]                      # 均衡子集
    # 标准化（用子集统计量；演示目的足够）后分层划分 训练 42 / 验证 18
    X_sub, _ = standard_scale(X_sub)
    X_tr, y_tr, X_va, y_va = train_test_split(X_sub, y_sub, test_size=0.3,
                                              shuffle=True, stratify=y_sub, seed=0)
    print(f"鸢尾花子集: 共 {n_total} 条（每类 {n_total // len(classes)} 条）"
          f"-> 训练 {len(X_tr)} / 验证 {len(X_va)}")
    return X_tr, y_tr, X_va, y_va, class_names


def sec_b_dropout():
    """对比大 MLP 有无 Dropout 时的过拟合程度（小数据 + 大模型必过拟合）。"""
    print("\n" + "=" * 60)
    print("小节 b：Dropout 正则效果（鸢尾花 60 样本 + MLP 4-64-3）")
    print("=" * 60)
    X_tr, y_tr, X_va, y_va, class_names = _iris_subset(60)   # 小样本数据集
    EPOCHS = 150                                             # 固定轮数便于对比曲线

    def _run(with_dropout):
        """训练一个带/不带 Dropout 的 4-64-3 MLP，返回 history。"""
        set_seed(42)                                         # 同起点：同初始化 + 同批顺序
        layers = [Linear(4, 64, name="fc1"), ReLU()]
        if with_dropout:                                     # Dropout 只在训练期生效
            layers.append(Dropout(p=0.5))
        layers += [Linear(64, 3, name="fc2")]
        model = Sequential(*layers)
        opt = Adam(model.params(), lr=0.01)
        trainer = Trainer(model, CrossEntropyLoss(), opt,
                          task="classification", seed=42)
        return trainer.train(X_tr, y_tr, X_val=X_va, y_val=y_va,
                             epochs=EPOCHS, batch_size=16, verbose=False), model

    hist_plain, model_plain = _run(False)                    # 无 Dropout：容易过拟合
    hist_drop, model_drop = _run(True)                       # 有 Dropout：正则化
    print(f"无 Dropout: 训练损失 {hist_plain['train_loss'][-1]:.4f} / "
          f"验证损失 {hist_plain['val_loss'][-1]:.4f} / 验证准确率 {hist_plain['val_acc'][-1]:.3f}")
    print(f"有 Dropout: 训练损失 {hist_drop['train_loss'][-1]:.4f} / "
          f"验证损失 {hist_drop['val_loss'][-1]:.4f} / 验证准确率 {hist_drop['val_acc'][-1]:.3f}")

    # ---------- 画 4 条曲线：两模型 train/val 损失 ----------
    curves = {
        "无Dropout-训练损失": hist_plain["train_loss"],
        "无Dropout-验证损失": hist_plain["val_loss"],
        "有Dropout-训练损失": hist_drop["train_loss"],
        "有Dropout-验证损失": hist_drop["val_loss"],
    }
    path = os.path.join(OUT_DIR, "Dropout正则效果.png")
    plot_lines(curves, xlabel="轮次 Epoch", ylabel="交叉熵损失",
               title="Dropout 正则效果：鸢尾花 60 样本训练大 MLP(4-64-3)", path=path)
    print(f"[已保存] {path}")
    conclusions.append(
        "b) Dropout 正则：仅 42 条训练样本训练 4-64-3 大网络，无 Dropout 时"
        "训练损失一路下降而验证损失先降后升（过拟合鸿沟明显）；加入 Dropout(0.5) 后"
        "训练损失整体更高（随机失活逼网络学冗余特征），但验证损失更低更稳，"
        "验证准确率不降反升/保持——以牺牲训练集拟合换取泛化能力。")
    return hist_plain, hist_drop


# ======================================================================
# 小节 c：LRScheduler 阶梯衰减 + plateau 自适应衰减
# ======================================================================
def sec_c_lr_scheduler():
    """演示 step 模式（确定性阶梯）与 plateau 模式（验证损失平台期自适应）的 lr 曲线。"""
    print("\n" + "=" * 60)
    print("小节 c：LRScheduler 学习率调度（step 阶梯衰减 + plateau 自适应衰减）")
    print("=" * 60)
    EPOCHS = 50

    # ---------- 1. step 模式：每 10 轮 lr *= 0.5 ----------
    holder = param(np.zeros(1), name="dummy")                # 用一个哑参数创建真实优化器
    opt_step = MBGD([holder], lr=0.1)                        # 优化器只需 .lr 属性即可被调度
    sched_step = LRScheduler(opt_step, mode="step", step_size=10, gamma=0.5)
    lr_step = [sched_step.step(epoch=ep) for ep in range(EPOCHS)]  # 逐轮推进记录 lr
    print(f"step 模式:   lr 轨迹(前20轮) = {lr_step[:20]}")

    # ---------- 2. plateau 模式：模拟"先下降后平台"的验证损失 ----------
    opt_plateau = MBGD([param(np.zeros(1), name="dummy2")], lr=0.1)
    sched_plateau = LRScheduler(opt_plateau, mode="plateau", patience=3, gamma=0.5)
    # 模拟验证损失：前 20 轮缓慢改善 -> 之后进入平台期（触发衰减）
    sim_val_loss = list(np.linspace(1.0, 0.55, 20)) + [0.55] * 30
    lr_plateau = [sched_plateau.step(val_loss=v) for v in sim_val_loss]
    print(f"plateau 模式: lr 轨迹(前20轮) = {np.round(lr_plateau[:20], 4)}")
    print(f"plateau 模式: 平台期后 lr 降至 {min(lr_plateau)}")

    # ---------- 3. 画 lr 变化曲线（对数轴更直观） ----------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(lr_step, "-o", ms=3, label="step 模式（每10轮 ×0.5）")
    ax.plot(lr_plateau, "-s", ms=3, label="plateau 模式（验证损失停滞3轮 ×0.5）")
    ax.set_xlabel("轮次 Epoch")
    ax.set_ylabel("学习率 lr")
    ax.set_yscale("log")                                     # 对数轴看衰减阶梯更清晰
    ax.set_title("LRScheduler 学习率调度曲线对比")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    path = os.path.join(OUT_DIR, "学习率调度.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] {path}")
    conclusions.append(
        "c) 学习率调度：step 模式按固定周期（每 10 轮）把 lr 乘 0.5，形成确定性阶梯，"
        "训练后期小步长帮助在极小值附近收敛；plateau 模式监视验证损失，"
        "连续 3 轮无改善才衰减（前 20 轮损失在改善时 lr 保持 0.1 不变，"
        "进入平台期后自动降为 0.05/0.025...），比 step 更省心、不浪费高学习率的黄金期。")
    return lr_step, lr_plateau


# ======================================================================
# 小节 d：EarlyStopping 早停演示
# ======================================================================
def sec_d_early_stopping():
    """在大模型过拟合训练中注入 EarlyStopping 回调，观察提前停止的轮次。"""
    print("\n" + "=" * 60)
    print("小节 d：EarlyStopping 早停（过拟合训练被提前终止）")
    print("=" * 60)
    X_tr, y_tr, X_va, y_va, _ = _iris_subset(60)             # 小样本 -> 易过拟合
    set_seed(42)
    model = Sequential(Linear(4, 64, name="fc1"), ReLU(),
                       Linear(64, 3, name="fc2"))            # 无 Dropout：必然过拟合
    opt = Adam(model.params(), lr=0.01)
    es = EarlyStopping(patience=8, min_delta=1e-4)           # 连续 8 轮验证损失无改善即停
    state_log = {"best_epoch": 0, "best_loss": float("inf")}  # 记录最优轮次供报告

    def _callback(state):
        """ Trainer 回调：先记录最优轮次，再交给 EarlyStopping 判断是否停止。"""
        if state["val_loss"] < state_log["best_loss"]:       # 出现新的最优验证损失
            state_log["best_loss"] = state["val_loss"]
            state_log["best_epoch"] = state["epoch"]
        return es(state["val_loss"])                         # True -> Trainer 提前停止

    trainer = Trainer(model, CrossEntropyLoss(), opt, task="classification", seed=42)
    history = trainer.train(X_tr, y_tr, X_val=X_va, y_val=y_va,
                            epochs=300, batch_size=16,       # 给足 300 轮，看实际停在哪
                            callbacks=[_callback], verbose=False)
    stopped_at = len(history["train_loss"])                  # 实际训练轮数
    print(f"设定轮数 300 -> 实际停止于第 {stopped_at} 轮")
    print(f"最优验证损失 {state_log['best_loss']:.4f} 出现在第 {state_log['best_epoch']} 轮")
    print(f"（最优之后又耐心等了 {stopped_at - state_log['best_epoch']} 轮无改善才停止）")

    # ---------- 画曲线：验证损失 + lr，标注最优/停止点 ----------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(history["train_loss"], "-", color="#1f77b4", label="训练损失")
    ax.plot(history["val_loss"], "-", color="#ff7f0e", label="验证损失")
    ax.axvline(state_log["best_epoch"] - 1, color="#2ca02c", ls="--",
               label=f"最优验证损失（第 {state_log['best_epoch']} 轮）")
    ax.axvline(stopped_at - 1, color="#d62728", ls=":",
               label=f"早停触发（第 {stopped_at} 轮）")
    ax.set_xlabel("轮次 Epoch")
    ax.set_ylabel("交叉熵损失")
    ax.set_title("EarlyStopping 早停演示：patience=8")
    ax.grid(alpha=0.3)
    ax.legend()
    path = os.path.join(OUT_DIR, "早停演示.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] {path}")
    conclusions.append(
        f"d) EarlyStopping：300 轮预算下，验证损失在第 {state_log['best_epoch']} 轮"
        f"达到最优 {state_log['best_loss']:.4f}，此后连续 8 轮无改善，训练于第 "
        f"{stopped_at} 轮被自动终止——节省 "
        f"{300 - stopped_at} 轮无效计算（约 {(300 - stopped_at) / 3:.0f}% 训练时间），"
        "并天然保留了接近最优的模型状态，是防过拟合 + 省时省钱的双重保险。")
    return stopped_at, state_log


# ======================================================================
# 小节 e：保存 / 加载往返一致性
# ======================================================================
def _predict(model, X):
    """对 Sequential 模型做一次评估模式推理，返回 numpy 预测。

    说明：Sequential 本身只负责"层组合 + 前向"，不提供 predict 接口；
    推理统一走"评估模式 + Node 包装 + 取 .data"，避免误启用 Dropout。
    """
    model.train_mode(False)                                  # 关闭 Dropout 等训练态行为
    return np.asarray(model(Node(X)).data)                   # 前向并取出原始 logits


def sec_e_save_load():
    """训练 -> save_model -> 全新模型 load_model -> 逐元素对比预测是否完全一致。"""
    print("\n" + "=" * 60)
    print("小节 e：save_model / load_model 往返一致性验证")
    print("=" * 60)
    X_tr, y_tr, X_va, y_va, _ = _iris_subset(60)             # 数据
    set_seed(42)
    model_a = Sequential(Linear(4, 16, name="fc1"), ReLU(),
                         Linear(16, 3, name="fc2"))          # 模型 A：正常训练
    opt = Adam(model_a.params(), lr=0.01)
    trainer = Trainer(model_a, CrossEntropyLoss(), opt, task="classification", seed=42)
    trainer.train(X_tr, y_tr, epochs=60, batch_size=16, verbose=False)
    print(f"模型 A 训练完成（参数量 {count_params(model_a)}），验证准确率 "
          f"{trainer.evaluate(X_va, y_va)[1]:.3f}")

    # ---------- 保存 -> 全新初始化的模型 B 加载 ----------
    npz_path = os.path.join(OUT_DIR, "model_demo.npz")       # 保存路径
    save_model(model_a, npz_path)                            # 参数写入 npz
    print(f"[已保存] {npz_path}")
    set_seed(7)                                              # 故意用不同种子初始化模型 B
    model_b = Sequential(Linear(4, 16, name="fc1"), ReLU(),
                         Linear(16, 3, name="fc2"))
    preds_before = _predict(model_b, X_va)                   # 加载前：B 的预测（明显不同）
    load_model(model_b, npz_path)                            # 按参数名对号入座原地加载
    preds_after_b = _predict(model_b, X_va)                  # 加载后：B 的预测
    preds_a = _predict(model_a, X_va)                        # 模型 A 的预测（基准）

    # ---------- 一致性断言 + 打印对比表 ----------
    max_diff = float(np.max(np.abs(preds_a - preds_after_b)))  # 逐元素最大绝对差
    print(f"加载前模型 B 与 A 预测最大差异: "
          f"{float(np.max(np.abs(preds_before - preds_a))):.4f}（加载前应明显不同）")
    print(f"加载后模型 B 与 A 预测最大差异: {max_diff:.2e}（应为 0 或机器精度）")
    assert max_diff == 0.0, "save/load 往返后预测不一致！"
    print("[通过] 保存->加载 往返后预测结果完全一致（max_diff = 0）")
    print("\n前 5 个样本预测 logits 对比（A=训练后原模型 / B=重新加载模型）：")
    print(f"{'样本':>4} | {'A 的 logits':>28} | {'B 的 logits':>28}")
    for i in range(5):
        a_row = np.round(preds_a[i], 3)
        b_row = np.round(preds_after_b[i], 3)
        print(f"{i:>4} | {str(a_row):>28} | {str(b_row):>28}")

    # ---------- 一致性可视化：A/B logits 散点应落在 y=x 直线上 ----------
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(preds_a.ravel(), preds_after_b.ravel(), s=18, color="#1f77b4",
               label="A 与 B 的预测值对")
    lims = [preds_a.min() - 0.5, preds_a.max() + 0.5]        # 参考线 y=x 范围
    ax.plot(lims, lims, "r--", lw=1, label="y = x（完全一致）")
    ax.set_xlabel("模型 A（原始）预测 logits")
    ax.set_ylabel("模型 B（load_model 后）预测 logits")
    ax.set_title(f"保存/加载往返一致性：max_diff = {max_diff:.1e}")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(OUT_DIR, "保存加载一致性.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] {path}")
    conclusions.append(
        "e) save_model / load_model：模型训练后参数存为 .npz（键=参数名），"
        "全新初始化的模型 B 加载后与模型 A 的全部预测 logits 逐元素最大差为 0，"
        "散点图所有点严格落在 y=x 参考线上——往返一致性（round-trip）通过，"
        "训练好的模型可安全落盘、复现与部署。")
    return max_diff


# ======================================================================
# 主流程
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("扩展⑤：框架优化功能演示（utils + optim + trainer 增强）")
    print("=" * 60)

    sec_a_optimizers()                                       # a) 优化器对比
    sec_b_dropout()                                          # b) Dropout 正则
    sec_c_lr_scheduler()                                     # c) 学习率调度
    sec_d_early_stopping()                                   # d) 早停
    sec_e_save_load()                                        # e) 保存/加载一致性

    # ---------- 汇总结论 md ----------
    md_path = os.path.join(OUT_DIR, "优化功能演示结论.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 扩展⑤ 框架优化功能演示结论\n\n")
        f.write("本报告由 experiments/exp5_optimizations.py 自动生成。\n\n")
        for line in conclusions:                             # 各小节结论逐条写入
            f.write(line + "\n\n")
    print(f"\n[已保存] {md_path}")
    print("\n" + "=" * 60)
    print("扩展⑤ 优化功能演示完成！全部图片输出至 outputs/exp5/")
    print("=" * 60)
