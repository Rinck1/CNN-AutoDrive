# -*- coding: utf-8 -*-
"""
mnist_web.py —— MNIST 训练 + Web 实时看板 + 预测服务（课程要求 1/3）
====================================================================

功能
----
1. 训练一个经典 CNN（可选 lenet / vgg11 / mobilenet / resnet）；
2. 训练过程中在浏览器实时查看：损失曲线、验证指标、学习率、
   权重分布直方图、每层梯度范数、遮挡敏感性热图；
3. 训练结束把模型保存为 npz，并通过 `/predict` 提供**预测服务**：
   POST JSON 图像 → 返回类别/概率/延迟；也可 `--load` 只部署不训练。

运行（项目根目录）
------------------
    python applications/mnist_web.py                     # LeNet + 看板（约 20 秒）
    python applications/mnist_web.py --model vgg11       # VGG11 风格
    python applications/mnist_web.py --model mobilenet --width 0.25
    python applications/mnist_web.py --load              # 只加载已保存模型并起服务
    python applications/mnist_web.py --epochs 1 --no-serve  # 训练完即退出（CI 用）

浏览器打开 http://127.0.0.1:8000/ 即可看到实时看板；
预测服务示例：

    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d "{\"image\": [[0.0, 0.3, ...], ...]}"

产物：
    outputs/web/mnist_<model>.npz     训练好的参数
    outputs/web/训练摘要.json          最终指标 + 模型规模/FLOPs
"""

# ==================== 标准脚本初始化头（契约第 1 节） ====================
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common_setup import init_env
init_env(seed=42)

# Web 看板里会开 HTTP 线程；torchvision 的 MKL 与 numpy 的 MKL 在
# 新线程中可能触发 "OMP: Error #15"。本脚本只做 numpy 训练，
# 因此强制走 idx.gz 数据路径，避免加载 torch。
os.environ.setdefault('MINIFLOWS_NO_TORCHVISION', '1')
# ==============================================================================

import argparse
import json

import numpy as np

from miniflows.layers import (Sequential, Conv2D, MaxPool2D, Flatten, Linear,
                              ReLU)
from miniflows.losses import CrossEntropyLoss
from miniflows.optim import Adam
from miniflows.trainer import Trainer
from miniflows.utils import save_model, load_model, count_params
from miniflows.data import load_mnist_arrays, random_shift_images
from miniflows.profiler import profile_model
from miniflows.webui import TrainingDashboard
from miniflows.classic_cnn import build_vgg11, build_mobilenet_v1
from miniflows.resnet import build_resnet

OUT_DIR = os.path.join(_ROOT, 'outputs', 'web')
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================================
# 一、模型
# ======================================================================
def build_lenet():
    """LeNet 风格（28×28 → 10 类，44,426 参数）。"""
    return Sequential(
        Conv2D(1, 6, 5), ReLU(), MaxPool2D(2, 2),
        Conv2D(6, 16, 5), ReLU(), MaxPool2D(2, 2),
        Flatten(), Linear(256, 120), ReLU(),
        Linear(120, 84), ReLU(), Linear(84, 10),
    )


def build_model_by_name(name, width=0.25):
    """按名字构造模型；VGG/MobileNet 需要 32×32（会先上采样 MNIST）。"""
    key = name.lower()
    if key == 'lenet':
        return build_lenet(), False
    if key == 'vgg11':
        return build_vgg11(num_classes=10, width=width), True
    if key in ('mobilenet', 'mobilenet_v1'):
        return build_mobilenet_v1(num_classes=10, width=width), True
    if key == 'resnet':
        return build_resnet(in_channels=1), True
    raise ValueError(f'未知模型: {name}')


def upsample(x, size=64):
    """把 28×28 最近邻放大到 size×size（给 VGG/MobileNet/ResNet 用）。"""
    n, c, h, w = x.shape
    idx = (np.arange(size) * h // size).clip(0, h - 1)
    y = x[:, :, idx][:, :, :, idx]
    return np.ascontiguousarray(y)


# ======================================================================
# 二、Loss 包装：把 batch 损失实时喂给看板
# ======================================================================
class LoggedLoss:
    """损失包装器：调用真实损失，并把标量值写入看板（用于 batch 级曲线）。"""

    def __init__(self, loss_fn, dashboard):
        self.loss_fn = loss_fn
        self.dashboard = dashboard

    def __call__(self, pred, target):
        """前向损失 + 记录（不改变计算图结构）。"""
        loss = self.loss_fn(pred, target)
        self.dashboard.log_batch(float(loss.data))
        return loss


# ======================================================================
# 三、主流程
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='MNIST 训练 + Web 看板 + 预测服务')
    parser.add_argument('--model', default='lenet',
                        choices=['lenet', 'vgg11', 'mobilenet', 'resnet'])
    parser.add_argument('--width', type=float, default=0.25,
                        help='VGG/MobileNet 的宽度系数')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--samples', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--load', action='store_true', help='只加载已保存模型并起服务')
    parser.add_argument('--no-serve', action='store_true', help='训练完不常驻服务')
    args = parser.parse_args()

    model, need_upsample = build_model_by_name(args.model, args.width)
    npz = os.path.join(OUT_DIR, f'mnist_{args.model}.npz')
    input_shape = (1, 64, 64) if need_upsample else (1, 28, 28)

    # ---------- 1. 数据 ----------
    X_tr, y_tr, X_te, y_te = load_mnist_arrays(
        subset_train=args.samples, subset_test=1500)
    if need_upsample:
        X_tr, X_te = upsample(X_tr), upsample(X_te)
    print(f'数据：训练 {X_tr.shape}，测试 {X_te.shape}')

    # ---------- 2. 看板（先启动，训练时就能看） ----------
    dash = TrainingDashboard(
        model=model, loss_fn=CrossEntropyLoss(), task='classification',
        class_names=[str(i) for i in range(10)], port=args.port,
        title=f'MiniFlows 实时看板 · MNIST · {args.model}')
    dash.set_sample(X_te[:1], y_te[:1])
    url = dash.start()
    print(f'请在浏览器打开：{url}')

    # ---------- 3. 训练（或加载） ----------
    if args.load:
        if not os.path.exists(npz):
            raise SystemExit(f'找不到 {npz}，请先不带 --load 运行一次')
        load_model(model, npz)
        print(f'已加载 {npz}')
        summary = {'model': args.model, 'loaded': True,
                   'params': count_params(model)}
    else:
        optimizer = Adam(model.params(), lr=args.lr)
        loss_fn = LoggedLoss(CrossEntropyLoss(), dash)
        trainer = Trainer(model, loss_fn, optimizer,
                          task='classification', seed=42)
        hist = trainer.train(X_tr, y_tr, X_val=X_te, y_val=y_te,
                             epochs=args.epochs, batch_size=args.batch_size,
                             callbacks=[dash.callback], verbose=True)
        test_loss, test_acc = trainer.evaluate(X_te, y_te)
        print(f'最终测试：loss={test_loss:.4f}  acc={test_acc:.4f}')

        save_model(model, npz)
        prof = profile_model(model, input_shape, batch=1, nn_name=args.model)
        summary = {'model': args.model, 'epochs': args.epochs,
                   'samples': int(args.samples), 'test_acc': float(test_acc),
                   'test_loss': float(test_loss), **prof}
        print(f'已保存模型：{npz}')
        print(f"模型规模：{prof['params']:,} 参数，{prof['flops_human']}，"
              f"单帧 {prof['latency_ms']:.2f} ms，激活 {prof['act_mb']:.2f} MB")

    # ---------- 4. 预测服务演示（同一个 HTTP 端口） ----------
    demo = dash.predict(X_te[0, 0])           # 用第一张测试图演示 /predict 的返回
    print(f"预测服务示例：POST {url}predict  ->  {demo}")
    with open(os.path.join(OUT_DIR, '训练摘要.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    # ---------- 5. 常驻服务 ----------
    if args.no_serve:
        dash.stop()
        print('已按 --no-serve 退出（不常驻服务）')
    else:
        print('训练结束，预测服务继续运行；Ctrl+C 退出')
        dash.serve_forever()


if __name__ == '__main__':
    main()
