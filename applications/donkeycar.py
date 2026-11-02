# -*- coding: utf-8 -*-
"""
donkeycar.py —— 基本任务④：DonkeyCar 自动驾驶（模拟数据集 + CNN 端到端转向回归）
====================================================================
任务背景：
    DonkeyCar 是一辆 1:16 的开源无人小车，自动驾驶的核心回路是
    “摄像头图像 → 卷积神经网络 → 转向角（-1 左转 ~ +1 右转）”。
    本课程无真车，因此用 numpy 逐像素**渲染合成赛道俯视灰度图**构造
    数据集：图中有深灰背景、浅色路面带和左右两条亮色边界线，道路中心线
    在图像底部相对画面中心有随机横向偏移，且整体带随机倾斜。

    【图像→标签的物理含义】
    - 底部偏移 offset_bottom ∈ [-18,18]：车辆当前相对道路中心线的横向偏差
      （车偏右 → 道路中心在画面里偏左，即 offset_bottom<0）；
    - 顶部偏移 = offset_bottom + tilt：道路延伸方向的“斜率”，对应前方弯道；
    - 标签转向角 = k·(顶部偏移-底部偏移) + k2·底部偏移，再归一化到 [-1,1]：
      前项让车头对准道路延伸方向（跟踪弯道），后项把车拉回道路中线
      （纠偏）。负=左转，正=右转。真车上该标签由人工遥控时的方向盘
      角度录制（行为克隆），本实验由几何公式直接生成。

模型（LeNet 风格 CNN，输出经 Tanh 压到 [-1,1]）：
    Conv2D(1→8,k5,s2) → ReLU → MaxPool2 → Conv2D(8→16,k3) → ReLU → MaxPool2
    → Flatten → Linear(576→64) → ReLU → Linear(64→1) → Tanh
    尺寸推导（img=64）：64 →(k5,s2) 30 →(pool2) 15 →(k3,s1) 13 →(pool2) 6
    → Flatten 16*6*6=576。
    MSELoss + Adam(lr=1e-3)，600 训练 / 120 测试，epochs=8，batch=32，
    目标：测试集角度域 MAE < 0.1（不达标则自动加训到 15 轮重试）。

另含（可选加分）**闭环仿真**：从道路中心出发，每步把“车辆局部俯视视图”
    喂给训练好的模型取转向角，按简化运动学（航向积分）移动车辆，画出
    闭环轨迹 vs 不转向开环轨迹的对比 —— 这是真车（DonkeyCar）上路前的
    仿真验证环节。

运行方式（在项目根目录）：
    python applications/donkeycar.py

输出产物（outputs/donkeycar/）：
    训练曲线.png / 预测对比.png / 样例预测.png / 闭环仿真.png
    donkeycar_model.npz
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

from miniflows.layers import (Sequential, Conv2D, MaxPool2D, Flatten,
                              Linear, ReLU, Tanh)
from miniflows.losses import MSELoss
from miniflows.optim import Adam
from miniflows.trainer import Trainer
from miniflows.visualization import plot_loss_curve, plot_images_grid
from miniflows.utils import save_model

OUT_DIR = os.path.join(_ROOT, 'outputs', 'donkeycar')  # 输出目录：outputs/donkeycar/
os.makedirs(OUT_DIR, exist_ok=True)                    # 自动创建输出目录

# ------------------------- 渲染与标签的物理常量 -------------------------
IMG_SIZE = 64            # 图像边长（俯视图，单位：像素，也近似“世界像素”）
BG_COLOR = 35.0          # 深灰背景灰度
ROAD_COLOR = 130.0       # 浅色路面带灰度（比背景亮）
LINE_COLOR = 235.0       # 左右边界线灰度（最亮，方便 CNN 识别道路边缘）
ROAD_HALF_W = 10         # 道路半宽（像素），道路总宽 20 像素
OFFSET_RANGE = 18.0      # 底部偏移范围 [-18,18]（相对画面中心的像素数）
TILT_RANGE = 20.0        # 顶部-底部倾斜范围 [-20,20]（覆盖闭环仿真的弯道斜率）
K_TILT = 0.03            # 转向对“道路斜率”的增益（跟踪弯道）
K_OFFSET = 0.02          # 转向对“横向偏差”的增益（拉回中线）
LABEL_SCALE = K_TILT * TILT_RANGE + K_OFFSET * OFFSET_RANGE  # 标签理论最大值 0.96，用于归一化


# ======================================================================
# 1. 赛道图像渲染（直接操作 np.ndarray 像素，不使用 plt 画图——快且无字体依赖）
# ======================================================================
def _render_from_row_offsets(row_offsets, img_size=IMG_SIZE):
    """按“每行道路中心相对画面中心的横向偏移”渲染一张赛道俯视灰度图。

    参数：
        row_offsets (np.ndarray): 形状 (H,)；第 r 行道路中心的横向偏移
            （单位像素，0=画面正中，负=偏左，正=偏右）；
        img_size (int): 图像边长（正方形）。

    返回：
        np.ndarray: (H, W) 灰度图，取值 {背景35, 路面130, 边界线235}。

    渲染思路（纯 ndarray 广播，无逐像素循环）：
        1) 计算每个像素到“本行道路中心列”的横向距离 d（(H,W) 矩阵）；
        2) d ≤ 半宽 → 路面带；|d - 半宽| ≤ 0.9 → 约 2 像素宽的亮色边界线；
        3) 其余保持深灰背景。中心/路面越出画面时布尔掩码自动越界安全。
    """
    H = W = img_size
    cols = np.arange(W, dtype=np.float64)[None, :]            # (1,W) 列坐标，广播用
    cx = img_size / 2.0 + np.asarray(row_offsets).reshape(-1, 1)  # (H,1) 每行道路中心的绝对列号
    d = np.abs(cols - cx)                                      # (H,W) 每像素到中心线的横向距离
    img = np.full((H, W), BG_COLOR)                            # 1) 深灰背景打底
    img[d <= ROAD_HALF_W] = ROAD_COLOR                         # 2) 浅色路面带
    img[np.abs(d - ROAD_HALF_W) <= 0.9] = LINE_COLOR           # 3) 左右两条亮色边界线
    return img


def render_track_view(bottom_offset, top_offset, img_size=IMG_SIZE):
    """渲染一张带倾斜的赛道俯视图（数据集生成入口）。

    参数：
        bottom_offset (float): 图像最底行道路中心的横向偏移（车辆当前位置偏差）；
        top_offset (float): 图像最顶行道路中心的横向偏移（= bottom + tilt）；
        img_size (int): 图像边长。

    返回：
        np.ndarray: (H, W) 灰度图。

    几何含义：道路中心线从“顶部偏移”线性变化到“底部偏移”，逐行线性插值，
    模拟车辆前视方向上道路的倾斜/弯道。
    """
    rows = np.arange(img_size)
    t = rows / (img_size - 1.0)                                # 归一化行参数：0=顶部 → 1=底部
    # 逐行线性插值：顶部 top_offset → 底部 bottom_offset
    row_offsets = top_offset + (bottom_offset - top_offset) * t
    return _render_from_row_offsets(row_offsets, img_size)


def dataset_label(bottom_offset, top_offset):
    """由道路几何偏移计算“应当执行的转向角”标签，归一化到 [-1,1]。

    物理含义（行为克隆的标签生成规则）：
        转向 = K_TILT × (顶部偏移 - 底部偏移)   # 项1：车头对准道路延伸方向（跟弯）
             + K_OFFSET × 底部偏移              # 项2：把车辆拉回道路中线（纠偏）

    注意方向约定：画面 x 轴向右为正；offset_bottom>0 表示道路中心出现在
    画面右侧 → 车辆应右转 → 标签为正。因此 负=左转，正=右转，与 DonkeyCar
    真车的 PWM 转向约定一致。

    参数：
        bottom_offset (float): 底部偏移；top_offset (float): 顶部偏移。

    返回：
        float: 转向角 ∈ [-1,1]。
    """
    raw = K_TILT * (top_offset - bottom_offset) + K_OFFSET * bottom_offset  # 原始转向量
    # 归一化：系数设计使 raw ∈ [-0.96,0.96]，除以理论最大值映射到 [-1,1]，clip 作数值安全兜底
    return float(np.clip(raw / LABEL_SCALE, -1.0, 1.0))


def generate_track_dataset(n_samples=600, img_size=IMG_SIZE, seed=42):
    """生成“赛道俯视图 → 转向角”模拟数据集。

    参数：
        n_samples (int): 样本总数（含训练+测试）；
        img_size (int): 图像边长；
        seed (int): 随机种子。

    返回：
        (X, y): X 为 (n,1,H,W) float32 灰度图（已除以 255 归一到 [0,1]）；
                y 为 (n,) float64 转向角 ∈ [-1,1]。
    """
    rng = np.random.RandomState(seed)                     # 独立随机源：数据与训练互不干扰
    X = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        bottom = rng.uniform(-OFFSET_RANGE, OFFSET_RANGE)          # 底部偏移（车辆横向偏差）
        tilt = rng.uniform(-TILT_RANGE, TILT_RANGE)                # 道路倾斜（前方弯道）
        img = render_track_view(bottom, bottom + tilt, img_size)   # 顶部偏移 = 底部 + 倾斜
        X[i, 0] = img / 255.0                                      # 像素归一到 [0,1]
        y[i] = dataset_label(bottom, bottom + tilt)                # 几何公式生成监督标签
    return X, y


# ======================================================================
# 2. 模型：LeNet 风格 CNN，输出转向角 ∈ [-1,1]
# ======================================================================
def build_model():
    """构建转向角回归 CNN（详细中文注释含尺寸推导）。

    尺寸推导（输入 (N,1,64,64)，conv 输出 = floor((H-K)/stride)+1）：
        Conv2D(1→8, k5, s2): floor((64-5)/2)+1 = 30  → (N,8,30,30)
        MaxPool2D(2,2)     : floor(30/2)      = 15  → (N,8,15,15)
        Conv2D(8→16, k3,s1): floor((15-3)/1)+1= 13  → (N,16,13,13)
        MaxPool2D(2,2)     : floor((13-2)/2)+1= 6   → (N,16,6,6)
        Flatten            : 16*6*6 = 576            → (N,576)
        Linear(576→64)+ReLU→ Linear(64→1) → Tanh    → (N,1) ∈ [-1,1]
    """
    return Sequential(
        Conv2D(1, 8, 5, stride=2),      # (N,1,64,64)  -> (N,8,30,30)  浅层提取边缘/线条
        ReLU(),
        MaxPool2D(2, 2),                # -> (N,8,15,15)          下采样，扩大感受野
        Conv2D(8, 16, 3),               # -> (N,16,13,13)         组合线条为道路形状特征
        ReLU(),
        MaxPool2D(2, 2),                # -> (N,16,6,6)
        Flatten(),                      # -> (N,576)
        Linear(576, 64),                # 全局特征 → 隐层
        ReLU(),
        Linear(64, 1),                  # 回归输出 1 维
        Tanh(),                         # 压缩到 [-1,1]：负=左转，正=右转
    )


# ======================================================================
# 3. （可选加分）闭环仿真：模型转向控制 vs 不转向开环
# ======================================================================
def closed_loop_demo(trainer, path):
    """DonkeyCar 上路前的仿真验证（简化运动学闭环）。

    设定：
        - 世界坐标系：x 向右、z 向前；赛道中心线 cx(z) = A·sin(2πz/λ)
          （A=12px，λ=300px），道路半宽 10px；
        - 车辆从道路中心 (0,0) 出发、朝向正前（θ=0）；
        - 每步：按当前位姿渲染一张 64×64 的“车体局部俯视视图”（与训练数据
          同一套渲染函数，保证分布一致）→ 模型输出转向角 δ ∈[-1,1]
          → 航向 θ += GAIN·δ·MAX_DTHETA（简化运动学：转向角积分成航向变化）
          → 沿航向前进 SPEED 像素；
        - 开环对照：不转向（θ 恒 0），车辆直线行驶，弯道处必然压出路面。

    参数：
        trainer (Trainer): 已训练好的回归 Trainer（用其 predict 推理）；
        path (str): 对比图保存路径。

    返回：
        dict: {'closed_max_dev': 闭环最大横向偏差, 'open_max_dev': 开环最大横向偏差}
    """
    AMP, LAMBDA = 12.0, 300.0            # 弯道幅度/波长（保证斜率 ≤0.26，落在训练分布内）
    SPEED, STEPS = 2.5, 100              # 每步前进 2.5px，共 100 步 ≈ 250px 行程
    GAIN, MAX_DTHETA = 1.0, 0.05         # 满舵 1 步最大转 0.05 rad

    def cx_world(z):
        """世界坐标下道路中心线的 x 值。"""
        return AMP * np.sin(2.0 * np.pi * z / LAMBDA)

    def simulate(use_model):
        """跑一遍轨迹仿真。use_model=False 时不转向（开环）。返回 (xs, zs, devs)。"""
        x, z, th = 0.0, 0.0, 0.0         # 初始：道路中心、朝向正前
        xs, zs, devs = [x], [z], [x - cx_world(z)]
        rows = np.arange(IMG_SIZE)
        s_ahead = (IMG_SIZE - 1) - rows  # 每行对应的前向距离：底行=0（车当前位置）→ 顶行=63
        for _ in range(STEPS):
            # ---- 由车辆位姿计算视图中每行道路中心的横向偏移（车体坐标系） ----
            wx = x + s_ahead * np.sin(th)            # 该行视点的世界 x（沿航向前伸）
            wz = z + s_ahead * np.cos(th)            # 该行视点的世界 z
            lateral = (cx_world(wz) - wx) * np.cos(th)  # 道路中心相对车体的横向偏差（俯视简化投影）
            view = _render_from_row_offsets(lateral, IMG_SIZE) / 255.0  # 与训练同款渲染，(64,64)
            if use_model:
                # 模型推理：输入 (1,1,H,W)，输出 (1,1) ∈ [-1,1]
                pred = trainer.predict(view[None, None, :, :])
                delta = float(np.ravel(pred)[0])
                th = float(np.clip(th + GAIN * delta * MAX_DTHETA, -0.6, 0.6))  # 航向积分并限幅
            x += SPEED * np.sin(th)                  # 简化运动学：沿航向匀速前进
            z += SPEED * np.cos(th)
            xs.append(x); zs.append(z)
            devs.append(x - cx_world(z))             # 横向偏差：车 - 道路中心
        return np.array(xs), np.array(zs), np.array(devs)

    xs_c, zs_c, dev_c = simulate(True)    # 闭环：模型转向
    xs_o, zs_o, dev_o = simulate(False)   # 开环：不转向对照
    print(f"闭环仿真：模型转向最大横向偏差 = {np.max(np.abs(dev_c)):.2f} px（半宽 {ROAD_HALF_W} px）")
    print(f"开环对照：不转向最大横向偏差   = {np.max(np.abs(dev_o)):.2f} px")

    # ---- 画对比图：路面带 + 中心线 + 两条轨迹 ----
    fig, ax = plt.subplots(figsize=(9, 8))
    z_grid = np.linspace(-5, zs_c.max() + 10, 400)          # 沿 z 采样道路带
    ax.fill_betweenx(z_grid, cx_world(z_grid) - ROAD_HALF_W, cx_world(z_grid) + ROAD_HALF_W,
                     color='#bbbbbb', alpha=0.7, label='赛道路面（半宽10px）')
    ax.plot(cx_world(z_grid), z_grid, 'k--', lw=1, label='道路中心线')
    ax.plot(xs_c, zs_c, 'r.-', ms=4, lw=1.5, label='模型闭环轨迹')
    ax.plot(xs_o, zs_o, 'b.--', ms=4, lw=1.2, label='不转向开环轨迹')
    ax.plot(0, 0, 'k*', ms=14, label='起点（道路中心）')
    ax.set_xlabel('横向 x（px）'); ax.set_ylabel('前进方向 z（px）')
    ax.set_title(f'DonkeyCar 闭环仿真验证：模型转向最大偏差 {np.max(np.abs(dev_c)):.1f}px ｜ '
                 f'开环不转向最大偏差 {np.max(np.abs(dev_o)):.1f}px（压出路面）')
    ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3); ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)              # Agg 后端：只保存
    print(f"闭环仿真图已保存：{path}")
    return dict(closed_max_dev=float(np.max(np.abs(dev_c))), open_max_dev=float(np.max(np.abs(dev_o))))


# ======================================================================
# 4. 主流程：生成数据 → 训练 → 评估 → 出图 → 存模型 → 闭环仿真
# ======================================================================
def main():
    """DonkeyCar 模拟实验主流程。"""
    print('=' * 70)
    print('DonkeyCar 自动驾驶（模拟）：CNN 端到端转向角回归')
    print('=' * 70)

    # ---------- 1. 生成并划分数据集：600 训练 / 120 测试 ----------
    X, y = generate_track_dataset(n_samples=720, img_size=IMG_SIZE, seed=42)  # 720 = 600+120
    perm = np.random.RandomState(0).permutation(len(X))      # 独立洗牌，保证训练/测试同分布
    X_tr, y_tr = X[perm[:600]], y[perm[:600]]
    X_te, y_te = X[perm[600:]], y[perm[600:]]
    y_tr_col = y_tr.reshape(-1, 1)                           # 标签整理成 (N,1)，与模型输出对齐，
    y_te_col = y_te.reshape(-1, 1)                           # 避免 (N,1)-(N,) 广播成 (N,N) 的错位
    print(f"训练集 {X_tr.shape[0]} 条，测试集 {X_te.shape[0]} 条，图像 {IMG_SIZE}×{IMG_SIZE}")

    # ---------- 2. 构建模型 / 优化器 / 训练器 ----------
    model = build_model()
    try:
        model.summary((1, IMG_SIZE, IMG_SIZE))               # 打印每层输出形状与参数量（中文表格）
    except Exception as exc:                                 # summary 为增强功能，缺省时降级打印
        print(f"(summary 不可用：{exc}) 参数量见模型定义注释")
    optimizer = Adam(model.params(), lr=1e-3)                # Adam：对学习率不敏感、收敛快
    trainer = Trainer(model, MSELoss(), optimizer, task='regression')

    # ---------- 3. 快速训练 8 轮（不达标自动加训到 15 轮） ----------
    EPOCHS_1 = 8
    hist1 = trainer.train(X_tr, y_tr_col, X_te, y_te_col,
                          epochs=EPOCHS_1, batch_size=32, verbose=True)
    pred = np.ravel(trainer.predict(X_te))                   # 测试集预测转向角
    mse_v = float(np.mean((pred - y_te) ** 2))               # 角度域 MSE
    mae_v = float(np.mean(np.abs(pred - y_te)))              # 角度域 MAE（目标 < 0.1）
    print(f"[{EPOCHS_1} 轮] 测试 MSE={mse_v:.5f}  MAE={mae_v:.4f}  "
          f"{'达标(目标<0.1)' if mae_v < 0.1 else '未达标，加训 7 轮…'}")

    hist_all = hist1
    if mae_v >= 0.1:
        # 兜底策略：Adam 状态延续（同一 optimizer 对象），再训 7 轮凑满 15；
        # 若仍不达标则如实报告（模拟数据上通常第 8 轮即可达标）。
        hist2 = trainer.train(X_tr, y_tr_col, X_te, y_te_col,
                              epochs=15 - EPOCHS_1, batch_size=32, verbose=True)
        pred = np.ravel(trainer.predict(X_te))
        mse_v = float(np.mean((pred - y_te) ** 2))
        mae_v = float(np.mean(np.abs(pred - y_te)))
        print(f"[15 轮] 测试 MSE={mse_v:.5f}  MAE={mae_v:.4f}  "
              f"{'达标(目标<0.1)' if mae_v < 0.1 else '仍未达标'}")
        # 合并两段 history 供画完整训练曲线（键不存在时安全跳过）
        hist_all = {k: (hist1.get(k, []) + hist2.get(k, [])) for k in hist1.keys()}

    # ---------- 4. 产物 1：训练损失曲线 ----------
    try:
        plot_loss_curve(hist_all, os.path.join(OUT_DIR, '训练曲线.png'),
                        title='DonkeyCar 转向角回归训练曲线（MSELoss + Adam）')
    except Exception:
        # 降级：histories 键名不一致时手动画训练损失
        plt.figure(figsize=(7, 4.5))
        plt.plot(hist_all.get('train_loss', []), label='训练损失')
        if hist_all.get('val_loss'):
            plt.plot(hist_all['val_loss'], '--', label='验证损失')
        plt.xlabel('轮次'); plt.ylabel('MSE'); plt.title('DonkeyCar 训练曲线')
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, '训练曲线.png'), dpi=150); plt.close()
    print("训练曲线已保存：outputs/donkeycar/训练曲线.png")

    # ---------- 5. 产物 2：预测 vs 真值散点图 ----------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, pred, s=18, alpha=0.7, edgecolors='k', linewidths=0.3)  # 每点=一个测试样本
    lim = [-1.1, 1.1]
    plt.plot(lim, lim, 'r--', lw=1, label='理想：预测=真值')  # 对角线参考
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel('真值转向角'); plt.ylabel('预测转向角')
    plt.title(f'DonkeyCar 预测 vs 真值（测试集）\nMSE={mse_v:.4f}  MAE={mae_v:.4f}（目标<0.1）')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '预测对比.png'), dpi=150); plt.close()
    print("预测对比图已保存：outputs/donkeycar/预测对比.png")

    # ---------- 6. 产物 3：8 张样例图（标题含真实/预测角度） ----------
    n_show = 8
    titles = [f"真:{y_te[i]:+.2f} 预:{pred[i]:+.2f}" for i in range(n_show)]  # 每张图下方标注
    try:
        plot_images_grid(X_te[:n_show], titles, os.path.join(OUT_DIR, '样例预测.png'), n=n_show, cols=4)
    except Exception as exc:
        # 降级：visualization.plot_images_grid 不可用时用 matplotlib 手绘
        print(f"(plot_images_grid 降级：{exc})")
        fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
        for ax, i in zip(axes.ravel(), range(n_show)):
            ax.imshow(X_te[i, 0], cmap='gray', vmin=0, vmax=1)
            ax.set_title(titles[i], fontsize=10); ax.axis('off')
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, '样例预测.png'), dpi=150); plt.close(fig)
    print("样例预测图已保存：outputs/donkeycar/样例预测.png")

    # ---------- 7. 产物 4：保存模型权重（npz，named_params 一一对应） ----------
    save_model(model, os.path.join(OUT_DIR, 'donkeycar_model.npz'))
    print("模型已保存：outputs/donkeycar/donkeycar_model.npz")

    # ---------- 8. 产物 5（可选加分）：闭环仿真 ----------
    closed_loop_demo(trainer, os.path.join(OUT_DIR, '闭环仿真.png'))

    # ---------- 9. 汇总打印 ----------
    print('\n' + '=' * 70)
    print(f'最终结果：测试 MSE = {mse_v:.5f}，MAE = {mae_v:.4f}（目标 < 0.1：'
          f"{'达标' if mae_v < 0.1 else '未达标'}）")
    print('=' * 70)


if __name__ == '__main__':
    main()
