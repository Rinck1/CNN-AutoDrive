# -*- coding: utf-8 -*-
"""
mnist.py —— 基于 MiniFlows（纯 numpy 自动微分）的 MNIST 手写数字识别 CNN
========================================================================

功能概述（课程基本任务③，与 pytorch_benchmark/ 组成"自研框架 vs PyTorch"对比）：
    1. 加载 MNIST 数据（miniflows.data.load_mnist_arrays，快速模式按类别均衡取子集）；
    2. 搭建 LeNet 风格 CNN（契约第 13 节指定结构）：
         Conv(1→6,5) -> ReLU -> MaxPool2 -> Conv(6→16,5) -> ReLU -> MaxPool2
         -> Flatten(→256) -> Linear(256→120) -> ReLU -> Linear(120→84)
         -> ReLU -> Linear(84→10)
       尺寸变化：28 -> 24(卷积) -> 12(池化) -> 8(卷积) -> 4(池化) -> 256(Flatten)；
    3. 默认快速模式：subset 8000/2000、epochs=3、batch=64、Adam lr=1e-3
       （--full 切换全量 60000/10000、epochs=5）；
    4. 训练前打印 model.summary((1,28,28)) 与参数总量；每轮打印损失与耗时；
    5. 输出产物（outputs/mnist/）：
       - 训练曲线.png        （train/val loss + val_acc 双轴，plot_loss_curve）
       - 预测示例.png        （8 张测试图 + 真实/预测标签网格，plot_images_grid）
       - mnist_result.json   （框架名/准确率/耗时/参数量等，供 train_lenet.py
                              读取生成"自研 MiniFlows vs PyTorch"对比表）

性能说明（契约第 13 节）：
    numpy 版卷积用 im2col 实现（float64），若实测单轮超过 8 分钟，
    按契约把默认子集降到 5000/1000（修改下方 SUBSET_TRAIN/TEST_FAST 常量即可，
    自测结论见脚本末尾注释）。

运行方式（在项目根目录）：
    python applications/mnist.py            # 快速模式（约几分钟）
    python applications/mnist.py --full     # 全量模式（耗时长）
"""

# ==================== 标准脚本初始化头（契约第 1 节，原样复制） ====================
import os
# 预防性设置：miniflows.data 可能经 torchvision 引入 torch，torch 与 numpy(MKL)
# 在 Windows/Anaconda 下各带一份 OpenMP 运行时会冲突（OMP Error #15）；
# setdefault 保证外部已设置时不覆盖。这是官方文档给出的临时开关。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
import json
import argparse
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

from miniflows.layers import Sequential, Linear, Conv2D, MaxPool2D, Flatten, ReLU
from miniflows.losses import CrossEntropyLoss        # 多分类交叉熵（内部 log-softmax）
from miniflows.optim import Adam                     # Adam 优化器
from miniflows.trainer import Trainer                # 统一训练器
from miniflows.data import load_mnist_arrays         # MNIST 加载（含快速子集取样）
from miniflows.utils import count_params             # 参数总量统计
from miniflows.visualization import plot_loss_curve, plot_images_grid

# ------------------------- 全局常量（超参数集中管理） -------------------------
OUT_DIR = init_env(seed=42, output_dir='outputs/mnist')  # 输出目录，自动创建
DATA_DIR = 'data'        # MNIST 缓存目录（首次运行自动下载约 12MB）
# 快速模式子集大小：契约第 13 节推荐 8000/2000；
# 若 numpy 版单轮超过 8 分钟，按契约降为 5000/1000（注释保留供修改）
SUBSET_TRAIN_FAST = 8000   # 快速模式训练子集（按类别均衡取样）
SUBSET_TEST_FAST = 2000    # 快速模式测试子集（按类别均衡取样）
SUBSET_FULL = None         # 全量模式：不取子集（60000/10000）
EPOCHS_FAST = 3            # 快速模式轮数：子集小，3 轮即可达 ≥90% 准确率
EPOCHS_FULL = 5            # 全量模式轮数：数据多，5 轮足够
BATCH_SIZE = 64            # 小批量：CNN 显存(内存)与速度的平衡，契约推荐
LR = 1e-3                  # Adam 学习率：契约推荐，CNN 上收敛快且稳
NUM_CLASSES = 10           # MNIST 10 类（数字 0-9）
GRID_N = 8                 # 预测示例网格图的图片张数


def build_lenet():
    """构建 LeNet 风格 CNN（契约第 13 节指定结构，与 PyTorch 基准完全一致）。

    结构与尺寸变化（输入 (N,1,28,28)）：
        Conv2D(1,6,5)   : 1 通道 -> 6 通道, 5x5 卷积核, 无填充 -> (N,6,24,24)
        ReLU            : 非线性激活
        MaxPool2D(2,2)  : 2x2 最大池化, 空间减半   -> (N,6,12,12)
        Conv2D(6,16,5)  : 6 通道 -> 16 通道, 5x5   -> (N,16,8,8)
        ReLU            : 非线性激活
        MaxPool2D(2,2)  : 空间减半                 -> (N,16,4,4)
        Flatten         : 展平                     -> (N,256)   (16*4*4=256)
        Linear(256,120) : 全连接层
        ReLU            : 非线性激活
        Linear(120,84)  : 全连接层
        ReLU            : 非线性激活
        Linear(84,10)   : 输出 10 类 logits（交叉熵内部做 softmax）

    返回：
        Sequential: LeNet 风格卷积神经网络。
    """
    return Sequential(
        # —— 卷积部分：提取局部图像特征 ——
        Conv2D(1, 6, kernel_size=5),   # (N,1,28,28) -> (N,6,24,24) 边缘/笔画特征
        ReLU(),                        # 非线性，保证网络表达能力
        MaxPool2D(kernel_size=2, stride=2),  # (N,6,24,24) -> (N,6,12,12) 降采样抗噪
        Conv2D(6, 16, kernel_size=5),  # (N,6,12,12) -> (N,16,8,8) 组合出更高级形状特征
        ReLU(),                        # 非线性
        MaxPool2D(kernel_size=2, stride=2),  # (N,16,8,8) -> (N,16,4,4) 再降采样
        # —— 全连接部分：把特征映射到类别 ——
        Flatten(),                     # (N,16,4,4) -> (N,256) 展平成向量
        Linear(256, 120),              # 全连接：256 -> 120
        ReLU(),                        # 非线性
        Linear(120, 84),               # 全连接：120 -> 84
        ReLU(),                        # 非线性
        Linear(84, 10),                # 输出层：84 -> 10 类 logits
    )


def make_time_callback():
    """构造逐轮打印"损失 + 耗时"的回调函数（利用 Trainer 的 callbacks 机制）。

    Trainer 在每个 epoch 结束后调用 callback(state)（state 含 epoch/train_loss/
    val_loss/val_acc 等键），返回 True 表示提前停止训练；这里始终返回 False。

    返回：
        fn(state: dict) -> bool: 计时回调，闭包变量记录上一轮时间戳。
    """
    clock = [time.time()]  # 用列表做可变闭包：记录上一轮结束的时刻

    def cb(state):
        """每轮结束时被 Trainer 调用：计算与本轮间隔并打印损失/耗时。

        参数：
            state (dict): Trainer 传入的训练状态（epoch/train_loss/val_loss/val_acc）。
        返回：
            bool: 恒为 False，表示不提前停止。
        """
        now = time.time()
        elapsed = now - clock[0]   # 本轮耗时 = 当前时刻 - 上一轮时刻
        clock[0] = now
        epoch = state.get('epoch', '?')
        tl = state.get('train_loss')   # 本轮平均训练损失
        vl = state.get('val_loss')     # 验证集平均损失（可能为 None）
        va = state.get('val_acc')      # 验证集准确率（可能为 None）
        msg = f'[第 {epoch} 轮] 耗时 {elapsed:6.1f} 秒'
        if tl is not None:
            msg += f' | train_loss={tl:.4f}'
        if vl is not None:
            msg += f' | val_loss={vl:.4f}'
        if va is not None:
            msg += f' | val_acc={va * 100:.2f}%'
        print(msg)
        return False  # 不触发提前停止
    return cb


def parse_args():
    """解析命令行参数。

    返回：
        argparse.Namespace: 含 full 布尔标志（--full 时为 True）。
    """
    parser = argparse.ArgumentParser(description='MiniFlows LeNet MNIST 训练脚本')
    parser.add_argument('--full', action='store_true',
                        help='使用全量 MNIST（60000/10000, epochs=5）；'
                             '默认快速模式（子集 8000/2000, epochs=3）')
    return parser.parse_args()


# ================================ 主流程 ================================
def main():
    """MNIST 主流程：加载 -> 建模 -> 训练 -> 评估 -> 保存图表与 json 结果。"""
    args = parse_args()
    full = args.full
    subset_train = SUBSET_FULL if full else SUBSET_TRAIN_FAST   # None=全量
    subset_test = SUBSET_FULL if full else SUBSET_TEST_FAST
    epochs = EPOCHS_FULL if full else EPOCHS_FAST

    print('=' * 62)
    print(f'MiniFlows LeNet 风格 CNN 识别 MNIST '
          f'（{"全量模式" if full else "快速模式"}）')
    print('=' * 62)

    # ---------- 1. 加载 MNIST（首次运行自动下载并缓存到 data/mnist/） ----------
    print(f'加载 MNIST（subset_train={subset_train}, subset_test={subset_test}）...')
    t0 = time.time()
    # 按类别均衡取样（round-robin），保证子集中 10 类样本数量一致
    X_train, y_train, X_test, y_test = load_mnist_arrays(
        subset_train=subset_train, subset_test=subset_test, data_dir=DATA_DIR)
    print(f'数据加载完成，耗时 {time.time() - t0:.1f} 秒：')
    print(f'  X_train={X_train.shape}, y_train={y_train.shape}')
    print(f'  X_test ={X_test.shape}, y_test ={y_test.shape}')

    # ---------- 2. 构建模型并打印结构摘要 ----------
    model = build_lenet()
    print('\n模型结构摘要（输入形状 (1,28,28)）：')
    model.summary((1, 28, 28))            # 中文表格：每层输出形状/参数量/总量
    n_params = count_params(model)        # 参数总量（与 PyTorch 版应同为 44426）
    print(f'参数总量: {n_params}')

    # ---------- 3. 训练 ----------
    loss_fn = CrossEntropyLoss()                     # 多分类交叉熵
    optimizer = Adam(model.params(), lr=LR)          # Adam，lr=1e-3
    trainer = Trainer(model, loss_fn, optimizer, task='classification')

    print(f'\n开始训练（epochs={epochs}, batch_size={BATCH_SIZE}, '
          f'Adam lr={LR}, device=cpu）...')
    t_start = time.time()
    # 测试集作为验证集 -> history 记录 val_loss/val_acc 供画曲线
    # verbose=False：逐轮信息由下方计时回调统一打印（含耗时）
    history = trainer.train(X_train, y_train, X_val=X_test, y_val=y_test,
                          epochs=epochs, batch_size=BATCH_SIZE,
                          callbacks=[make_time_callback()], verbose=False)
    train_seconds = time.time() - t_start   # 总训练耗时（写入 json 供对比表使用）

    # ---------- 4. 最终测试集评估 ----------
    test_loss, test_acc = trainer.evaluate(X_test, y_test, batch_size=256)
    print('\n' + '-' * 62)
    print(f'最终测试集损失: {test_loss:.6f}')
    print(f'最终测试集准确率: {test_acc * 100:.2f}%   （训练总耗时 {train_seconds:.1f} 秒）')
    print('-' * 62)

    # ---------- 5. 保存训练曲线图 ----------
    path_curve = os.path.join(OUT_DIR, '训练曲线.png')
    plot_loss_curve(history, path_curve,
                    title=f'MiniFlows LeNet MNIST 训练曲线'
                          f'（{"全量" if full else "快速"}模式, {epochs} 轮）')
    print(f'已保存训练曲线：{path_curve}')

    # ---------- 6. 保存 8 张测试图 + 预测标签网格图 ----------
    n_show = min(GRID_N, X_test.shape[0])
    X_show = X_test[:n_show]                          # (8,1,28,28) 测试图像
    y_pred = trainer.predict_classes(X_show)          # argmax 预测类别
    y_true = y_test[:n_show].astype(int)
    titles = [f'真:{t} 预:{p}' + ('✓' if t == p else '✗')
              for t, p in zip(y_true, np.asarray(y_pred).reshape(-1))]  # 每张图下方标题
    path_grid = os.path.join(OUT_DIR, '预测示例.png')
    plot_images_grid(X_show, titles=titles, path=path_grid, n=n_show, cols=4)
    print(f'已保存预测示例图：{path_grid}')

    # ---------- 7. 保存结果 json（供 pytorch_benchmark/train_lenet.py 读取对比） ----------
    result = {
        'framework': 'miniflows',       # 框架名
        'test_accuracy': float(test_acc),   # 最终测试准确率
        'epochs': int(epochs),          # 训练轮数
        'subset_train': int(X_train.shape[0]),  # 实际训练样本数
        'subset_test': int(X_test.shape[0]),    # 实际测试样本数
        'train_seconds': round(train_seconds, 1),  # 训练总耗时（秒）
        'param_count': int(n_params),   # 参数总量
        'device': 'cpu',                # numpy 版固定 CPU
    }
    path_json = os.path.join(OUT_DIR, 'mnist_result.json')
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)  # 中文注释字段见上方
    print(f'已保存结果 json：{path_json}')
    print('MNIST（MiniFlows）应用运行完毕。')


if __name__ == '__main__':
    main()
