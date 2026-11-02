# -*- coding: utf-8 -*-
"""
lenet5.py —— PyTorch 版 LeNet-5 卷积神经网络（对比基准）
========================================================

用途：
    与 applications/mnist.py 中 MiniFlows 版网络**结构完全一致**（契约第 13 节），
    保证"自研 MiniFlows vs PyTorch"对比的公平性。两者参数总量均为 61706。

网络结构（输入 (N,1,28,28)，MNIST 单通道 28x28）：
    Conv2d(1→6, kernel=5)   : 28×28 → 24×24（无填充，输出边长 = 28-5+1 = 24）
    ReLU                    : 非线性激活
    MaxPool2d(2)            : 24×24 → 12×12（2×2 窗口、步长 2，空间减半）
    Conv2d(6→16, kernel=5)  : 12×12 → 8×8（输出边长 = 12-5+1 = 8）
    ReLU                    : 非线性激活
    MaxPool2d(2)            : 8×8 → 4×4（空间再减半）
    Flatten                 : (N,16,4,4) → (N,256)（16×4×4 = 256）
    Linear(256→120) → ReLU  : 全连接
    Linear(120→84)  → ReLU  : 全连接
    Linear(84→10)           : 输出 10 类 logits（配合 nn.CrossEntropyLoss）

参数量明细（28×28 输入，Flatten 后为 16×4×4=256）：
    conv1: 6×(1×5×5)+6 = 156
    conv2: 16×(6×5×5)+16 = 2416
    fc1  : 256×120+120 = 30840
    fc2  : 120×84+84 = 10164
    fc3  : 84×10+10 = 850
    合计  = 44426（与 MiniFlows 版一致；经典论文的 61706 是 32×32 输入、
            Flatten 后 400→120 的版本，此处输入 28×28 故为 44426）

本文件为纯模块（被 train_lenet.py import），直接运行时打印结构摘要用于自检。
"""

# ==================== 标准脚本初始化头（契约第 1 节，原样复制） ====================
import os
# 解决 Windows/Anaconda 下 numpy(MKL) 与 torch 各自带一份 OpenMP 运行时
# （libiomp5md.dll）重复初始化导致的致命冲突（OMP Error #15）；
# setdefault 保证外部已设置时不覆盖。这是官方文档给出的临时开关。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
try:  # Windows 控制台强制 UTF-8，避免中文乱码报错
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # 使直接运行本文件时也能定位项目根目录
from common_setup import init_env
init_env(seed=42)  # 统一设置：随机种子 + matplotlib 中文字体 + Agg 后端
# ==============================================================================

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """PyTorch 版 LeNet-5（结构与 MiniFlows 版逐层一致，保证公平对比）。

    属性：
        features (nn.Sequential): 卷积 + 激活 + 池化特征提取部分；
        classifier (nn.Sequential): 展平 + 全连接分类部分。

    说明：
        构造函数中每一层都附有中文注释，标明卷积核大小与特征图尺寸变化
        （28→24→12→8→4），与 applications/mnist.py 的 build_lenet() 一一对应。
    """

    def __init__(self, num_classes=10):
        """初始化网络各层。

        参数：
            num_classes (int): 分类类别数，MNIST 为 10（数字 0-9）。
        """
        super().__init__()
        # ---------------- 卷积特征提取部分 ----------------
        self.features = nn.Sequential(
            # 卷积层1：输入 1 通道（灰度图），输出 6 通道，5×5 卷积核，无填充
            # 尺寸变化：28×28 → 24×24（边长 = 28 - 5 + 1 = 24），提取边缘/笔画特征
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            # ReLU 激活：引入非线性，负值置 0
            nn.ReLU(),
            # 最大池化1：2×2 窗口、步长 2，尺寸变化 24×24 → 12×12，降采样增强平移鲁棒性
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层2：输入 6 通道，输出 16 通道，5×5 卷积核，无填充
            # 尺寸变化：12×12 → 8×8（边长 = 12 - 5 + 1 = 8），组合出更高层形状特征
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            # ReLU 激活
            nn.ReLU(),
            # 最大池化2：2×2 窗口、步长 2，尺寸变化 8×8 → 4×4
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # ---------------- 全连接分类部分 ----------------
        self.classifier = nn.Sequential(
            # 展平层：(N,16,4,4) → (N,256)，256 = 16×4×4
            nn.Flatten(),
            # 全连接层1：256 → 120（与 MiniFlows 版 Linear(256,120) 一致）
            nn.Linear(in_features=256, out_features=120),
            # ReLU 激活
            nn.ReLU(),
            # 全连接层2：120 → 84（与 MiniFlows 版 Linear(120,84) 一致）
            nn.Linear(in_features=120, out_features=84),
            # ReLU 激活
            nn.ReLU(),
            # 输出层：84 → 10 类 logits（与 MiniFlows 版 Linear(84,10) 一致）；
            # 训练时配合 nn.CrossEntropyLoss（内部含 log-softmax），无需显式 Softmax
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        """前向传播。

        参数：
            x (torch.Tensor): 输入图像批次，形状 (N,1,28,28)，取值已归一化到 [0,1]。

        返回：
            torch.Tensor: 10 类 logits，形状 (N,10)。
        """
        out = self.features(x)       # 卷积部分：(N,1,28,28) → (N,16,4,4)
        out = self.classifier(out)   # 展平 + 全连接：(N,16,4,4) → (N,10)
        return out


def count_parameters(model):
    """统计模型可训练参数总量。

    参数：
        model (nn.Module): PyTorch 模型。

    返回：
        int: 所有 requires_grad=True 参数的元素个数之和。
    """
    # numel() 为参数张量元素数；仅统计需要求导的可训练参数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # ---- 直接运行本文件时的自检：打印结构摘要与参数总量 ----
    net = LeNet5()
    print('PyTorch LeNet-5 结构摘要（输入 (1,28,28)）：')
    y = torch.randn(1, 1, 28, 28)  # 构造一个假输入逐层验证尺寸
    print(f'  输入: {tuple(y.shape)}')
    print(f'  features 输出: {tuple(net.features(y).shape)}')       # 期望 (1,16,4,4)
    print(f'  classifier 输出: {tuple(net.classifier(net.features(y)).shape)}')  # 期望 (1,10)
    print(f'参数总量: {count_parameters(net)}')  # 期望 44426，与 MiniFlows 版一致
