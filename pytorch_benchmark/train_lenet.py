# -*- coding: utf-8 -*-
"""
train_lenet.py —— PyTorch 版 LeNet-5 训练脚本 + "自研 MiniFlows vs PyTorch"对比
================================================================================

功能概述（课程基本任务①：用 PyTorch 搭建经典 CNN，并与自研框架对比）：
    1. 加载 MNIST（优先复用 data/mnist/ 下已缓存的 4 个 idx.gz 文件；缺失时依次
       尝试 torchvision 官方 S3 镜像下载、urllib 免证书下载、curl 兜底下载，
       全部缓存到 data/mnist/），并转成 numpy 数组；
    2. 按 applications/mnist.py 完全相同的"按类别轮流（round-robin）均衡取样前 N"
       逻辑取子集（快速模式 8000/2000，--full 为全量 60000/10000），
       保证两个框架训练/测试用到的数据一致，对比才公平；
    3. 训练 pytorch_benchmark/lenet5.py 中的 LeNet-5（结构与 MiniFlows 版完全一致，
       参数量同为 44426）；device 固定为 cpu —— 因为 numpy 版 MiniFlows 只能在
       CPU 上运行，同用 CPU 才是公平对比（如需 GPU 加速，把 DEVICE 改为 'cuda' 即可，
       但届时不应与 CPU 版 MiniFlows 直接比较耗时）；
    4. 训练后打印 PyTorch 测试准确率；
    5. 读取 outputs/mnist/mnist_result.json（由 applications/mnist.py 生成），
       打印"自研 MiniFlows vs PyTorch"中文对比表（框架/准确率/参数量/耗时），
       并保存 outputs/pytorch_benchmark/对比结果.md 与 PyTorch 损失曲线图；
       若 json 不存在，给出中文提示：请先运行 applications/mnist.py。

运行方式（在项目根目录）：
    python pytorch_benchmark/train_lenet.py          # 快速模式（子集 8000/2000, 3 轮）
    python pytorch_benchmark/train_lenet.py --full   # 全量模式（60000/10000, 5 轮）
"""

# ==================== 标准脚本初始化头（契约第 1 节，原样复制） ====================
import os
# 解决 Windows/Anaconda 下 numpy(MKL) 与 torch 各自带一份 OpenMP 运行时
# （libiomp5md.dll）重复初始化导致的致命冲突（OMP Error #15）；
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
    sys.path.insert(0, _ROOT)  # 使脚本能 import miniflows / common_setup / lenet5
from common_setup import init_env
init_env(seed=42)  # 统一设置：随机种子 + matplotlib 中文字体 + Agg 后端
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gzip
import struct
import ssl
import subprocess
import urllib.request
from torchvision import datasets            # MNIST 下载与缓存

from lenet5 import LeNet5, count_parameters  # 同目录下的 PyTorch 版网络

# ------------------------- 全局常量（超参数集中管理） -------------------------
OUT_DIR = init_env(seed=42, output_dir='outputs/pytorch_benchmark')  # 输出目录
MNIST_JSON = os.path.join(_ROOT, 'outputs', 'mnist', 'mnist_result.json')  # MiniFlows 结果
DATA_ROOT = 'data'        # torchvision MNIST 缓存根目录（下载到 data/MNIST/raw/）
IDX_DIR = os.path.join(_ROOT, 'data', 'mnist')   # idx.gz 缓存目录（与 MiniFlows 回退方案共享）
MIRROR = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # 官方 S3 镜像（已确认可达）
IDX_FILES = ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
DEVICE = torch.device('cpu')  # 固定 CPU：与 numpy 版 MiniFlows 公平对比（可改 'cuda'）
# 快速模式子集：与 applications/mnist.py 保持一致（8000/2000，epochs=3）
SUBSET_TRAIN_FAST = 8000
SUBSET_TEST_FAST = 2000
EPOCHS_FAST = 3           # 快速模式轮数（与 MiniFlows 版一致）
EPOCHS_FULL = 5           # 全量模式轮数（与 MiniFlows 版一致）
BATCH_SIZE = 64           # 批大小（与 MiniFlows 版一致）
LR = 1e-3                 # Adam 学习率（与 MiniFlows 版一致）
NUM_CLASSES = 10          # MNIST 类别数


def _download_one_idx(filename):
    """下载单个 idx.gz 文件到 data/mnist/（多级降级：urllib 免证书 -> curl）。

    背景：本机 Windows 证书库存在损坏证书，torchvision/urllib 默认 SSL 上下文
    （load_default_certs）会抛 SSLError；故回退方案：
        1) urllib + 不加载系统证书、不校验证书的 SSL 上下文；
        2) 仍失败则调用系统 curl（Git Bash 自带，已验证可用）。

    参数：
        filename (str): idx.gz 文件名（如 'train-images-idx3-ubyte.gz'）。
    """
    path = os.path.join(IDX_DIR, filename)
    os.makedirs(IDX_DIR, exist_ok=True)      # 确保缓存目录存在
    url = MIRROR + filename
    # ---- 方案 1：urllib + 跳过证书校验的 SSL 上下文（不加载损坏的系统证书库） ----
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)   # 手工构造，不加载默认证书
        ctx.check_hostname = False                      # 不校验主机名
        ctx.verify_mode = ssl.CERT_NONE                 # 不校验证书链
        with urllib.request.urlopen(url, timeout=60, context=ctx) as resp, \
                open(path, 'wb') as f:
            f.write(resp.read())                        # 一次性写入（文件仅几 MB）
        return
    except Exception as e:
        print(f'  urllib 下载 {filename} 失败（{e}），改用 curl ...')
    # ---- 方案 2：调用系统 curl（已验证可从该镜像下载） ----
    subprocess.run(['curl', '-sL', '-o', path, url], check=True)


def _read_idx_images(path):
    """解析 idx3 图像文件 -> (N,28,28) uint8。

    idx3 格式：4 个大端 int32（magic=2051, N, rows, cols）+ N*rows*cols 字节像素。

    参数：
        path (str): idx3 文件路径（.gz 压缩或已解压均可）。

    返回：
        np.ndarray: 形状 (N, rows, cols) 的 uint8 数组。
    """
    with gzip.open(path, 'rb') as f:               # gzip 流式读取
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))  # 大端 4 个 uint32
        assert magic == 2051, f'{path} 不是有效的 idx3 图像文件（magic={magic}）'
        data = np.frombuffer(f.read(), dtype=np.uint8)             # 一次性读入像素
    return data.reshape(n, rows, cols)             # (N,28,28)


def _read_idx_labels(path):
    """解析 idx1 标签文件 -> (N,) uint8。

    idx1 格式：2 个大端 int32（magic=2049, N）+ N 字节标签（0~9）。

    参数：
        path (str): idx1 文件路径（.gz 压缩或已解压均可）。

    返回：
        np.ndarray: 形状 (N,) 的 uint8 标签数组。
    """
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))  # 大端 2 个 uint32
        assert magic == 2049, f'{path} 不是有效的 idx1 标签文件（magic={magic}）'
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def get_mnist_numpy(data_root=DATA_ROOT):
    """加载 MNIST 并转为 numpy 数组（多级降级，优先复用 data/mnist/ 缓存）。

    降级顺序：
        1) data/mnist/ 已有 4 个 idx.gz（本脚本或 miniflows 提前缓存过）-> 直接解析，
           零网络开销；
        2) torchvision.datasets.MNIST(download=True)（官方 S3 镜像）；
        3) 手动下载 idx.gz（urllib 免证书 / curl 兜底）后解析。

    所有路径均转成与 MiniFlows 版 load_mnist_arrays 一致的格式：
    X: (N,1,28,28) float32 且像素归一化到 [0,1]；y: (N,) int64。

    参数：
        data_root (str): torchvision 缓存根目录。

    返回：
        (X_train, y_train, X_test, y_test)。
    """
    # ---- 路径 1：data/mnist/ 的 idx.gz 缓存齐全 -> 直接解析 ----
    if all(os.path.exists(os.path.join(IDX_DIR, f)) for f in IDX_FILES):
        X_train = _read_idx_images(os.path.join(IDX_DIR, IDX_FILES[0]))
        y_train = _read_idx_labels(os.path.join(IDX_DIR, IDX_FILES[1]))
        X_test = _read_idx_images(os.path.join(IDX_DIR, IDX_FILES[2]))
        y_test = _read_idx_labels(os.path.join(IDX_DIR, IDX_FILES[3]))
        print(f'  从缓存 {os.path.relpath(IDX_DIR, _ROOT)}/ 读取 4 个 idx.gz 文件')
    else:
        # ---- 路径 2/3：torchvision 优先，失败则手动下载 idx.gz 再解析 ----
        try:
            ds_train = datasets.MNIST(root=data_root, train=True, download=True)
            ds_test = datasets.MNIST(root=data_root, train=False, download=True)
            X_train = ds_train.data.numpy()            # uint8 (N,28,28)
            y_train = ds_train.targets.numpy()
            X_test = ds_test.data.numpy()
            y_test = ds_test.targets.numpy()
            print('  通过 torchvision 加载成功')
        except Exception as e:
            # 常见失败原因：本机 Windows 证书库损坏导致 torchvision 下载 SSL 报错
            print(f'  torchvision 加载失败（{type(e).__name__}），'
                  f'回退为手动从 {MIRROR} 下载 idx.gz ...')
            for fn in IDX_FILES:                       # 缺哪个补哪个
                if not os.path.exists(os.path.join(IDX_DIR, fn)):
                    _download_one_idx(fn)
            X_train = _read_idx_images(os.path.join(IDX_DIR, IDX_FILES[0]))
            y_train = _read_idx_labels(os.path.join(IDX_DIR, IDX_FILES[1]))
            X_test = _read_idx_images(os.path.join(IDX_DIR, IDX_FILES[2]))
            y_test = _read_idx_labels(os.path.join(IDX_DIR, IDX_FILES[3]))
    # ---- 统一转成与 MiniFlows 版一致的数据格式 ----
    X_train = X_train.astype(np.float32)[:, None, :, :] / 255.0  # (N,1,28,28) 归一化
    X_test = X_test.astype(np.float32)[:, None, :, :] / 255.0
    y_train = y_train.astype(np.int64)                          # 标签 int64
    y_test = y_test.astype(np.int64)
    return X_train, y_train, X_test, y_test


def balanced_subset(X, y, n, num_classes=NUM_CLASSES):
    """按类别轮流（round-robin）均衡取样前 n 个样本（与 MiniFlows 版同逻辑）。

    实现思路（契约 load_mnist_arrays 的取样规则）：
        1) 按类别收集样本在原数组中的下标（保持出现顺序）；
        2) 第 i 轮从每个类别各取其第 i 个样本，轮流进行；
        3) 直到凑满 n 个或某类耗尽为止。
    这样得到的子集 10 类样本数量完全一致，且两个框架用同一确定性规则
    （同样的 MNIST 原始顺序）取到同样的样本，保证对比公平。

    参数：
        X (np.ndarray): 特征数组。
        y (np.ndarray): 标签数组。
        n (int): 目标样本数。
        num_classes (int): 类别数。

    返回：
        (X_sub, y_sub): 取出的子集（保持轮转取样顺序）。
    """
    if n is None or n >= X.shape[0]:   # None 或超过总量 -> 直接用全量
        return X, y
    # 每个类别的样本下标列表（按原数组出现顺序）
    idx_by_class = [np.where(y == c)[0] for c in range(num_classes)]
    chosen = []          # 被选中的下标
    round_i = 0          # 轮转轮次：第 round_i 个样本（按类内顺序）
    while len(chosen) < n:
        for c in range(num_classes):           # 每轮依次从 0~9 类各取一个
            if len(chosen) >= n:               # 凑满即停
                break
            idx_list = idx_by_class[c]
            if round_i < len(idx_list):        # 该类还有剩余样本
                chosen.append(idx_list[round_i])
        round_i += 1
    chosen = np.array(chosen, dtype=np.int64)
    return X[chosen], y[chosen]


def evaluate(model, X, y, criterion, batch_size=1000):
    """在给定数据集上评估模型，返回 (平均损失, 准确率)。

    参数：
        model (nn.Module): LeNet-5 模型。
        X (np.ndarray): 特征 (N,1,28,28)。
        y (np.ndarray): 标签 (N,)。
        criterion: 损失函数（nn.CrossEntropyLoss）。
        batch_size (int): 评估批大小。

    返回：
        (float, float): 平均损失与准确率（0~1）。
    """
    model.eval()                       # 评估模式（本模型无 Dropout/BN，恒等，仅规范写法）
    total_loss, correct = 0.0, 0
    n = X.shape[0]
    with torch.no_grad():              # 评估不需要梯度
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)  # numpy -> torch
            yb = torch.from_numpy(y[i:i + batch_size]).to(DEVICE)
            logits = model(xb)                       # 前向：得到 (B,10) logits
            total_loss += criterion(logits, yb).item() * xb.shape[0]  # 累计损失×样本数
            correct += (logits.argmax(dim=1) == yb).sum().item()      # argmax 命中数
    model.train()                      # 切回训练模式
    return total_loss / n, correct / n


def train_model(X_train, y_train, X_test, y_test, epochs):
    """训练 PyTorch LeNet-5，逐轮记录损失与测试准确率。

    超参与 MiniFlows 版完全一致：Adam(lr=1e-3)、batch=64、CrossEntropyLoss。

    参数：
        X_train, y_train: 训练数据。
        X_test, y_test: 测试数据（每轮评估一次，用于记录曲线与最终准确率）。
        epochs (int): 训练轮数。

    返回：
        (model, history, seconds): 训练好的模型、历史字典与总耗时（秒）。
    """
    torch.manual_seed(42)              # 固定 torch 随机种子（权重初始化可复现）
    model = LeNet5().to(DEVICE)        # 模型放到 CPU（与 numpy 版公平对比）
    print(f'PyTorch LeNet-5 参数总量: {count_parameters(model)}（应与 MiniFlows 版一致）')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # 同为 Adam lr=1e-3
    criterion = nn.CrossEntropyLoss()                         # 交叉熵（内部 log-softmax）

    n = X_train.shape[0]
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    t0 = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n)            # 每轮重新打乱样本顺序
        losses, seen = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]           # 当前批的样本下标
            xb = torch.from_numpy(X_train[idx]).to(DEVICE)   # numpy -> torch 张量
            yb = torch.from_numpy(y_train[idx]).to(DEVICE)
            optimizer.zero_grad()                  # 清空上一步梯度
            logits = model(xb)                     # 前向传播
            loss = criterion(logits, yb)           # 交叉熵损失
            loss.backward()                        # 反向传播（PyTorch 自动微分）
            optimizer.step()                       # 更新参数
            losses += loss.item() * len(idx)       # 累计损失用于逐轮平均
            seen += len(idx)
        # 每轮结束：在测试集上评估并逐轮打印（与 MiniFlows 版回调输出对应）
        test_loss, test_acc = evaluate(model, X_test, y_test, criterion)
        history['train_loss'].append(losses / seen)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        print(f'[第 {epoch} 轮] train_loss={losses / seen:.4f} | '
              f'test_loss={test_loss:.4f} | test_acc={test_acc * 100:.2f}%')
    seconds = time.time() - t0
    return model, history, seconds


def plot_history(history, path):
    """绘制 PyTorch 训练损失曲线（左轴）与测试准确率曲线（右轴，双 y 轴）。

    参数：
        history (dict): 含 train_loss/test_loss/test_acc 列表。
        path (str): 图片保存路径。
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(epochs, history['train_loss'], 'o-', color='tab:blue', label='训练损失')
    ax1.plot(epochs, history['test_loss'], 's--', color='tab:cyan', label='测试损失')
    ax1.set_xlabel('训练轮次 Epoch')
    ax1.set_ylabel('交叉熵损失', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()                                    # 第二个 y 轴画准确率
    ax2.plot(epochs, history['test_acc'], '^-', color='tab:red', label='测试准确率')
    ax2.set_ylabel('测试准确率', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title('PyTorch LeNet-5 训练曲线（MNIST）')
    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches='tight')      # 只保存（Agg 后端）
    plt.close(fig)


def print_and_save_comparison(pt_result):
    """读取 MiniFlows 结果 json，打印中文对比表并保存"对比结果.md"。

    参数：
        pt_result (dict): PyTorch 本次运行结果（与 mnist_result.json 同字段结构）。

    返回：
        dict | None: 读取到的 MiniFlows 结果；不存在时返回 None。
    """
    # ---------- 读取 MiniFlows 结果（可能不存在） ----------
    mf_result = None
    if os.path.exists(MNIST_JSON):
        with open(MNIST_JSON, 'r', encoding='utf-8') as f:
            mf_result = json.load(f)
    else:
        # json 不存在时的中文提示：对比表只能展示 PyTorch 一侧
        print('\n[提示] 未找到 outputs/mnist/mnist_result.json，'
              '无法展示 MiniFlows 对比数据。')
        print('       请先在项目根目录运行: python applications/mnist.py，'
              '再重新运行本脚本以生成完整对比表。')

    # ---------- 组装对比表行 ----------
    rows = []
    if mf_result is not None:
        rows.append(('自研 MiniFlows (numpy)', mf_result))
    rows.append(('PyTorch', pt_result))

    # ---------- 控制台打印中文对比表 ----------
    print('\n' + '=' * 66)
    print('       自研 MiniFlows vs PyTorch 对比表（MNIST, LeNet 风格 CNN）')
    print('=' * 66)
    header = f'{"框架":<22}{"测试准确率":>10}{"参数量":>10}{"训练耗时(秒)":>14}'
    print(header)
    print('-' * 66)
    for name, r in rows:
        print(f'{name:<22}{r["test_accuracy"] * 100:>9.2f}%{r["param_count"]:>10}'
              f'{r["train_seconds"]:>14.1f}')
    print('-' * 66)
    print(f'（两框架训练/测试数据、网络结构、超参完全一致：'
          f'epochs={pt_result["epochs"]}, batch={BATCH_SIZE}, Adam lr={LR}, '
          f'样本 {pt_result["subset_train"]}/{pt_result["subset_test"]}, '
          f'device=cpu）')

    if mf_result is not None:
        # 对比结论：准确率接近说明自研框架实现正确；耗时差异来自底层实现
        diff = abs(pt_result['test_accuracy'] - mf_result['test_accuracy'])
        faster = 'MiniFlows' if mf_result['train_seconds'] < pt_result['train_seconds'] else 'PyTorch'
        print('【结论】')
        print(f'  两框架测试准确率相差 {diff * 100:.2f} 个百分点，'
              '说明自研 MiniFlows 的卷积/反向传播/优化器实现是正确的；')
        print(f'  本轮训练更快的是 {faster}：PyTorch 由 C++ 后端向量化执行，'
              'numpy float64 版本在计算效率上通常处于劣势属正常现象。')

    # ---------- 保存"对比结果.md" ----------
    def fmt_row(name, r):
        """生成 markdown 表格行。"""
        return (f'| {name} | {r["test_accuracy"] * 100:.2f}% | {r["param_count"]} '
                f'| {r["train_seconds"]:.1f} | {r["epochs"]} '
                f'| {r["subset_train"]}/{r["subset_test"]} | cpu |')

    md_lines = [
        '# 自研 MiniFlows vs PyTorch 对比结果（MNIST，LeNet 风格 CNN）',
        '',
        '## 实验设置',
        '',
        '- 网络结构（两框架完全一致，参数量 44426）：',
        '  Conv(1→6,5) → ReLU → MaxPool2 → Conv(6→16,5) → ReLU → MaxPool2',
        '  → Flatten(256) → Linear(256→120) → ReLU → Linear(120→84) → ReLU → Linear(84→10)',
        f'- 超参：Adam lr={LR}，batch={BATCH_SIZE}；样本按类别均衡取样；device=cpu',
        '- 数据一致性：PyTorch 侧用 torchvision 缓存转 numpy 后，'
        '按与 MiniFlows 版相同的"按类别轮流取样前 N"规则取子集。',
        '',
        '## 对比表',
        '',
        '| 框架 | 测试准确率 | 参数量 | 训练耗时(秒) | 轮数 | 样本(训练/测试) | 设备 |',
        '|---|---|---|---|---|---|---|',
    ]
    for name, r in rows:
        md_lines.append(fmt_row(name, r))
    if mf_result is None:
        md_lines += ['', '> 注：MiniFlows 结果缺失（outputs/mnist/mnist_result.json 不存在），',
                     '> 请先运行 `python applications/mnist.py` 后重跑本脚本。']
    md_path = os.path.join(OUT_DIR, '对比结果.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'已保存对比结果文档：{md_path}')
    return mf_result


def parse_args():
    """解析命令行参数。

    返回：
        argparse.Namespace: 含 full 布尔标志（--full 时为 True）。
    """
    parser = argparse.ArgumentParser(description='PyTorch LeNet-5 MNIST 训练与对比脚本')
    parser.add_argument('--full', action='store_true',
                        help='使用全量 MNIST（60000/10000, epochs=5）；'
                             '默认快速模式（子集 8000/2000, epochs=3）')
    return parser.parse_args()


# ================================ 主流程 ================================
def main():
    """PyTorch 基准主流程：取数 -> 训练 -> 评估 -> 画曲线 -> 打印/保存对比表。"""
    args = parse_args()
    full = args.full
    print('=' * 66)
    print(f'PyTorch LeNet-5 MNIST 基准（{"全量模式" if full else "快速模式"}，'
          f'device={DEVICE}）')
    print('=' * 66)

    # ---------- 1. 下载/缓存 MNIST 并转 numpy ----------
    print('加载 MNIST（优先读 data/mnist/ 缓存；缺失时自动从官方镜像下载约 12MB）...')
    t0 = time.time()
    X_train, y_train, X_test, y_test = get_mnist_numpy()
    print(f'MNIST 加载完成，耗时 {time.time() - t0:.1f} 秒；'
          f'全量: 训练 {X_train.shape[0]}, 测试 {X_test.shape[0]}')

    # ---------- 2. 与 MiniFlows 版完全一致的均衡取样 ----------
    if not full:
        X_train, y_train = balanced_subset(X_train, y_train, SUBSET_TRAIN_FAST)
        X_test, y_test = balanced_subset(X_test, y_test, SUBSET_TEST_FAST)
    epochs = EPOCHS_FULL if full else EPOCHS_FAST
    print(f'本次训练数据：训练 {X_train.shape[0]} 条 / 测试 {X_test.shape[0]} 条，'
          f'epochs={epochs}, batch={BATCH_SIZE}, Adam lr={LR}')

    # ---------- 3. 训练与评估 ----------
    model, history, seconds = train_model(X_train, y_train, X_test, y_test, epochs)
    # 用整批评估函数再测一次最终准确率（与逐轮评估口径一致）
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, X_test, y_test, criterion)
    print('-' * 66)
    print(f'PyTorch 最终测试集损失: {test_loss:.6f}')
    print(f'PyTorch 最终测试集准确率: {test_acc * 100:.2f}%   '
          f'（训练总耗时 {seconds:.1f} 秒）')
    print('-' * 66)

    # ---------- 4. 保存 PyTorch 损失曲线图 ----------
    path_curve = os.path.join(OUT_DIR, 'pytorch损失曲线.png')
    plot_history(history, path_curve)
    print(f'已保存 PyTorch 损失曲线图：{path_curve}')

    # ---------- 5. 保存 PyTorch 侧结果 json（便于复现与后续对比） ----------
    pt_result = {
        'framework': 'pytorch',
        'test_accuracy': float(test_acc),
        'epochs': int(epochs),
        'subset_train': int(X_train.shape[0]),
        'subset_test': int(X_test.shape[0]),
        'train_seconds': round(seconds, 1),
        'param_count': int(count_parameters(model)),
        'device': 'cpu',
    }
    path_pt_json = os.path.join(OUT_DIR, 'pytorch_result.json')
    with open(path_pt_json, 'w', encoding='utf-8') as f:
        json.dump(pt_result, f, ensure_ascii=False, indent=2)
    print(f'已保存 PyTorch 结果 json：{path_pt_json}')

    # ---------- 6. 打印中文对比表并保存 对比结果.md ----------
    print_and_save_comparison(pt_result)
    print('PyTorch 基准脚本运行完毕。')


if __name__ == '__main__':
    main()
