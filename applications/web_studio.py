# -*- coding: utf-8 -*-
"""
applications/web_studio.py —— MiniFlows 全功能交互式 Web 平台
=============================================================

把整个 MiniFlows 课程项目打包为一个零第三方 Web 依赖（仅 Python 标准库
http.server + SSE + 自包含 Canvas/SVG 前端）的一体化交互式工作台，涵盖：

1. **一键训练与实时可视化看板**：
   - 支持全部 5 类任务：线性回归、逻辑回归、鸢尾花分类、MNIST（LeNet / VGG11 /
     MobileNetV1 / ResNet-GN）、DonkeyCar 自动驾驶（ResNet-GN 主模型 / CNN / ViT-Tiny）；
   - 支持调节优化器（Adam / MBGD）、学习率、轮数、batch、早停、学习率调度、
     全局梯度裁剪（max_norm）、L2 正则、动量、数据标准化与 DonkeyCar 3× 数据增强；
   - SSE 实时推送 Train/Val 损失、Batch 细粒度损失、准确率/MAE、学习率、
     各层权重直方图、各层梯度 L2 范数、线性拟合线 / 逻辑边界 / 混淆矩阵及日志；
   - 支持一键保存模型（.npz）、一键加载已有权重（含 outputs/ 下预训练模型），
     并自动执行保存/加载往返预测零误差检验（diff）。
2. **在线推理、手绘板、赛道交互与闭环仿真**：
   - MNIST 28×28 交互式手写画板 → 实时类别概率、延迟与 8×8 遮挡热图；
   - DonkeyCar 赛道视角实时渲染器（11 项视觉扰动：偏移/弯道斜率/噪声/亮度/对比度/
     模糊/遮挡）→ 方向盘仪表盘预测转向角、8×8 遮挡热图与真实闭环行驶轨迹对比仿真；
   - **11 项视觉扰动鲁棒性一键扫描**：固定测试集逐项扰动，输出各扰动 MAE 与最差项排名。
3. **框架原理与四大扩展实验室**：
   - 扩展① 计算图自动构建与高清 DAG 渲染（x*x+x / MLP / ResNet BasicBlock）；
   - 扩展② 17 项算子与 MLP 中心差分数值梯度检验台（eps=1e-5）；
   - 扩展③ 四类异常训练与修复实时对比（梯度爆炸+裁剪 / 梯度消失 / 学习率震荡 / 过拟合）；
   - 扩展④ 鸢尾花 z-score / minmax 特征标准化无泄露对比。
4. **训练运行历史与实验报告导出**：
   - 每次训练结束自动落盘（配置 + 逐轮曲线 + 指标 + 参数量 + 耗时），
     支持多轮曲线叠加对比、明细表逐条删除；
   - 一键导出实验报告：Markdown 正文，或 ZIP（Markdown + charts/*.png 图表）。
5. **多架构对比、PyTorch 基准与主模型选型总览**：
   - 自动汇总 outputs/ 中跨框架资源对比、CNN vs ViT-Tiny vs ResNet-GN 同协议对比
     及调参消融结果与可视化图库。

运行方式（项目根目录）：
    python applications/web_studio.py [--port 8000]
    python web_app.py [--port 8000]      # 等价的快捷入口
"""

# ==================== 标准脚本初始化头（契约第 1 节） ====================
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass

_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FROZEN = bool(getattr(sys, 'frozen', False))
# PyInstaller 打包后资源在 sys._MEIPASS；源码运行时就是项目根目录
_ROOT = getattr(sys, '_MEIPASS', _SRC_ROOT) if _FROZEN else _SRC_ROOT
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common_setup import init_env
init_env(seed=42)

# Web 服务开多线程，强制禁用 torchvision 防止 MKL/OMP 与 NumPy 线程冲突
os.environ.setdefault('MINIFLOWS_NO_TORCHVISION', '1')
# ==============================================================================

import argparse
import base64
import io
import json
import math
import mimetypes
import threading
import time
import traceback
import zipfile
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import matplotlib.pyplot as plt

from miniflows import ops
from miniflows.graph import Node, constant
from miniflows.layers import (
    Sequential, Linear, Conv2D, MaxPool2D, Flatten, ReLU, Sigmoid, Tanh, Dropout
)
from miniflows.losses import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss, l2_penalty
from miniflows.optim import Adam, MBGD
from miniflows.autodiff import gradient_check, check_model_gradients
from miniflows.profiler import profile_model, count_params
from miniflows.visualization import draw_computation_graph
from miniflows.data import (
    load_iris, load_mnist_arrays, make_blobs, train_test_split,
    standard_scale, minmax_scale, apply_scale, random_shift_images, DataLoader
)
from miniflows.utils import (
    set_seed, accuracy, mse, EarlyStopping, LRScheduler, save_model, load_model
)
from miniflows.resnet import ResNetRegressor, BasicBlock, build_resnet
from miniflows.classic_cnn import build_vgg11, build_mobilenet_v1
from miniflows.transformer import build_vit

from applications.donkeycar import (
    generate_track_dataset, build_model as build_donkeycar_cnn,
    render_track_view, _render_from_row_offsets, dataset_label,
    IMG_SIZE as DC_IMG_SIZE, ROAD_HALF_W
)
from applications.donkeycar_eval import (
    perturb_images, regression_metrics, predict_array,
    SPEED, STEPS, DTHETA, MAX_HEADING, FAIL_DEV, TRAIN_FRAC, VAL_FRAC,
)
from applications.inspector import (
    trace_architecture, trace_backward, layer_layout, weight_pointcloud,
)

DATA_DIR = os.path.join(_ROOT, 'data')
if _FROZEN:
    # 单文件 exe 解压目录是临时的，训练产物/保存模型写到 exe 同级的 outputs
    _APP_DIR = os.path.dirname(sys.executable)
    OUTPUTS_DIR = os.path.join(_APP_DIR, 'outputs')
    _BUNDLED_OUTPUTS = os.path.join(_ROOT, 'outputs')
    if (not os.path.exists(os.path.join(OUTPUTS_DIR, 'donkeycar_resnet', 'donkeycar_resnet_model.npz'))
            and os.path.isdir(_BUNDLED_OUTPUTS)):
        try:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            shutil.copytree(_BUNDLED_OUTPUTS, OUTPUTS_DIR, dirs_exist_ok=True)
        except Exception:
            pass
else:
    OUTPUTS_DIR = os.path.join(_ROOT, 'outputs')
STUDIO_OUT_DIR = os.path.join(OUTPUTS_DIR, 'web_studio')
os.makedirs(STUDIO_OUT_DIR, exist_ok=True)

RUNS_FILE = os.path.join(STUDIO_OUT_DIR, 'runs.json')   # 训练运行历史（跨重启保留）
MAX_RUNS = 30                                          # 最多保留的运行条数

_PLOT_LOCK = threading.Lock()

# 前端扰动下拉框用的英文键 → donkeycar_eval.perturb_images 认的中文 kind
# （11 项视觉扰动全集，与 ROBUSTNESS_KINDS 一一对应）
DISTURB_KINDS = {
    'clean': 'clean',
    'noise_0.02': '噪声σ=0.02',
    'noise_0.05': '噪声σ=0.05',
    'noise_0.10': '噪声σ=0.10',
    'bright_-0.2': '亮度-0.2',
    'bright_+0.2': '亮度+0.2',
    'contrast_0.7': '对比度×0.7',
    'contrast_1.3': '对比度×1.3',
    'blur_3x3': '模糊3×3',
    'occ_center': '中心遮挡',
    'occ_bottom': '底部遮挡',
}


# ======================================================================
# 一、适配分类头的 ResNet-GN（复用 ResNetRegressor 主干）
# ======================================================================
class ResNetClassifier(ResNetRegressor):
    """把 ResNet-GN 回归主干改为多分类头（去掉 Tanh，输出 logits）。"""

    def __init__(self, in_channels=1, num_classes=10, c0=8, widths=(8, 16, 32),
                 blocks=(2, 2, 2), num_groups=None, name='resnet_cls'):
        super().__init__(in_channels=in_channels, c0=c0, widths=widths,
                         blocks=blocks, num_groups=num_groups, name=name)
        self.num_classes = int(num_classes)
        self.head = Linear(self.widths[-1], self.num_classes, name='head')

    def forward(self, x):
        """x:(N,C,H,W) -> (N,num_classes) logits。"""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = ops.mean(x, axis=(2, 3))
        return self.head(x)


def _upsample_64(x, size=64):
    """将 (N,1,H,W) 图像最近邻上采样到 (N,1,size,size)。"""
    arr = np.asarray(x, dtype=np.float64)
    _, _, h, _ = arr.shape
    if h == size:
        return arr
    idx = (np.arange(size) * h // size).clip(0, h - 1)
    return np.ascontiguousarray(arr[:, :, idx][:, :, :, idx])


def _softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def _clip_grad_norm(params, max_norm):
    """全局 L2 梯度范数裁剪，返回裁剪前总范数。"""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float(np.sum(p.grad ** 2))
    total_norm = float(np.sqrt(total_sq))
    if max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return total_norm


def _augment_donkeycar(X, y, seed=42):
    """3× 轻量数据增强（仅作用于训练集）：原图 + 光度变化 + 轻微横向微调。"""
    rng = np.random.RandomState(seed)
    X_aug = [X]
    y_aug = [y]
    # 第 2 份：亮度与对比度扰动（标签不变）
    alpha = rng.uniform(0.8, 1.2, size=(len(X), 1, 1, 1))
    beta = rng.uniform(-0.12, 0.12, size=(len(X), 1, 1, 1))
    X_photo = np.clip((X - 0.5) * alpha + 0.5 + beta, 0.0, 1.0)
    X_aug.append(X_photo)
    y_aug.append(y.copy())
    # 第 3 份：轻量高斯噪声 + 小幅随机平移
    X_shift = random_shift_images(X, max_shift=1, seed=seed + 7)
    X_noisy = np.clip(X_shift + rng.normal(0.0, 0.025, size=X_shift.shape), 0.0, 1.0)
    X_aug.append(X_noisy)
    y_aug.append(y.copy())
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


def _f(v):
    if v is None:
        return None
    val = float(v)
    return None if (np.isnan(val) or np.isinf(val)) else round(val, 6)


def _downsample_2d(img2d, size=28):
    h, w = img2d.shape
    if h == size and w == size:
        return img2d
    yi = (np.arange(size) * h // size).clip(0, h - 1)
    xi = (np.arange(size) * w // size).clip(0, w - 1)
    return img2d[yi][:, xi]


# ======================================================================
# 二、Studio 核心状态与一键训练调度器
# ======================================================================
class MiniFlowsStudio:
    """线程安全的 MiniFlows 交互式全功能控制器。"""

    def __init__(self, host='127.0.0.1', port=8000):
        self.host = host
        self.port = int(port)
        self._lock = threading.RLock()
        self._clients = []
        self._server = None
        self._http_thread = None
        self._train_thread = None
        self._stop_server = threading.Event()
        self._stop_train = threading.Event()
        self._stop_live = threading.Event()
        self._stop_drive = threading.Event()
        self._live = None            # 实时测试集评估状态
        self._drive = None           # 8 agent 实时驾驶状态
        self._live_thread = None
        self._drive_thread = None

        # 当前激活模型与元信息
        self.status = 'idle'  # idle | training | completed | stopped | error
        self.task = 'donkeycar'
        self.arch = 'resnet_gn'
        self.model = None
        self.loss_fn = None
        self.scaler = None
        self.need_upsample = False
        self.class_names = None
        self.input_shape = (1, 64, 64)
        self.model_info = {'params': 0, 'flops': 0, 'latency_ms': 0.0, 'tag': '未训练/未加载'}

        # 训练曲线与实时统计
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_metric': [], 'lr': []}
        self.batch_loss = deque(maxlen=400)
        self.batch_step = deque(maxlen=400)
        self._step = 0
        self.grad_norms = []
        self.param_hists = []
        self.logs = deque(maxlen=150)
        self.task_preview = {}   # 任务专属实时预览（拟合线/边界网格/混淆矩阵）
        self.sample_x = None
        self.sample_y = None
        self.last_saved_path = None
        # 训练运行历史（每次训练结束落盘一条，用于多曲线对比与报告导出）
        self.runs = self._load_runs()
        self._run_seq = max([int(r.get('id', 0)) for r in self.runs], default=0)
        self.robustness = None      # 最近一次 11 项扰动扫描结果
        self.epochs_total = 0
        self.progress = 0.0
        self.train_started = None
        self.started_at = time.time()

        # 启动时尝试预载已有主模型（若存在），让推理面板开箱即用
        self._preload_default_model()

    def _log(self, msg):
        ts = time.strftime('%H:%M:%S')
        with self._lock:
            self.logs.append(f'[{ts}] {msg}')

    # ------------------------------------------------------------------
    # 训练运行历史：落盘 / 读取 / 清理
    # ------------------------------------------------------------------
    def _load_runs(self):
        """读取历史运行记录；文件损坏时优雅退化为空列表。"""
        try:
            with open(RUNS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _persist_runs(self):
        """把运行历史写回磁盘（失败只记录日志，不影响训练主流程）。"""
        try:
            tmp = RUNS_FILE + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.runs, f, ensure_ascii=False, indent=1)
            os.replace(tmp, RUNS_FILE)
        except Exception as exc:
            self._log(f'运行历史写入失败：{exc}')

    def list_runs(self):
        """返回运行历史（按时间倒序，最新在前）。"""
        with self._lock:
            return list(reversed(self.runs))

    def clear_runs(self):
        with self._lock:
            self.runs = []
            self._persist_runs()
        self._log('已清空训练运行历史。')
        return {'ok': True, 'count': 0}

    def delete_run(self, run_id):
        with self._lock:
            before = len(self.runs)
            self.runs = [r for r in self.runs if int(r.get('id', -1)) != int(run_id)]
            removed = before - len(self.runs)
            self._persist_runs()
        return {'ok': True, 'removed': removed}

    def _record_run(self, cfg, bundle, status, elapsed_s, final_metric, best_metric,
                    metric_name, prof, log_msg):
        """把一次训练的结果追加到运行历史（供多曲线对比与报告导出）。"""
        with self._lock:
            self._run_seq += 1
            run = {
                'id': self._run_seq,
                'task': bundle['task'],
                'arch': bundle['arch'],
                'label': f"{bundle['task']} · {bundle['arch']}",
                'cfg': {
                    k: cfg.get(k) for k in
                    ('epochs', 'samples', 'lr', 'batch_size', 'optimizer', 'clip_norm',
                     'l2_reg', 'scale', 'seed', 'augment', 'early_stop', 'lr_schedule')
                },
                'epoch': list(self.history['epoch']),
                'train_loss': list(self.history['train_loss']),
                'val_loss': list(self.history['val_loss']),
                'val_metric': list(self.history['val_metric']),
                'lr': list(self.history['lr']),
                'metric_name': metric_name,
                'params': int(prof['params']),
                'flops': int(prof['flops']),
                'latency_ms': round(float(prof['latency_ms']), 3),
                'elapsed_s': round(float(elapsed_s), 2),
                'final_metric': final_metric,
                'best_metric': best_metric,
                'status': status,
                'n_train': int(len(bundle['X_tr'])),
                'n_val': int(len(bundle['X_va'])),
                'finished_at': time.time(),
                'finished_at_str': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            self.runs.append(run)
            if len(self.runs) > MAX_RUNS:
                self.runs = self.runs[-MAX_RUNS:]
            self._persist_runs()
        self._log(f'已记录运行 #{run["id"]}（{metric_name} {final_metric:.4f}，{elapsed_s:.1f}s）。{log_msg}')
        return run

    def _preload_default_model(self):
        resnet_npz = os.path.join(OUTPUTS_DIR, 'donkeycar_resnet', 'donkeycar_resnet_model.npz')
        if os.path.exists(resnet_npz):
            try:
                m = build_resnet()
                load_model(m, resnet_npz)
                m.train_mode(False)
                self.task = 'donkeycar'
                self.arch = 'resnet_gn'
                self.model = m
                self.loss_fn = MSELoss()
                self.input_shape = (1, 64, 64)
                self.need_upsample = False
                self.class_names = None
                prof = profile_model(m, self.input_shape)
                self.model_info = {
                    'params': int(prof['params']),
                    'flops': int(prof['flops']),
                    'latency_ms': round(float(prof['latency_ms']), 3),
                    'tag': '预载主模型 ResNet-GN (donkeycar_resnet_model.npz)',
                }
                img = render_track_view(6.0, 14.0, DC_IMG_SIZE) / 255.0
                self.sample_x = img[None, None, :, :].astype(np.float64)
                self.sample_y = np.array([dataset_label(6.0, 14.0)])
                self.last_saved_path = resnet_npz
                self._refresh_stats_locked()
                self._log('已自动预载 DonkeyCar 核心主模型 ResNet-GN（44,337 参数），可直接测试推理与闭环仿真。')
                return
            except Exception:
                pass
        # 若无预训练文件，也构造默认未训练样本用于展示
        img = render_track_view(6.0, 14.0, DC_IMG_SIZE) / 255.0
        self.sample_x = img[None, None, :, :].astype(np.float64)
        self._log('MiniFlows Web Studio 已就绪，可选择任意任务与架构一键训练。')

    # ------------------------------------------------------------------
    # 模型构造与数据准备
    # ------------------------------------------------------------------
    def _create_model(self, task, arch, cfg=None):
        """统一模型骨架工厂：保证训练与加载时参数命名 100% 一致。"""
        cfg = cfg or {}
        if task == 'linear':
            return Sequential(Linear(1, 1)), MSELoss(), (1,), False, 'regression', None
        if task == 'logistic':
            return Sequential(Linear(2, 1)), BCEWithLogitsLoss(), (2,), False, 'classification', ['类别 0', '类别 1']
        if task == 'iris':
            return (
                Sequential(Linear(4, 16), ReLU(), Linear(16, 3)),
                CrossEntropyLoss(), (4,), False, 'classification', ['setosa', 'versicolor', 'virginica']
            )
        if task == 'mnist':
            if arch == 'lenet':
                m = Sequential(
                    Conv2D(1, 6, 5), ReLU(), MaxPool2D(2, 2),
                    Conv2D(6, 16, 5), ReLU(), MaxPool2D(2, 2),
                    Flatten(), Linear(256, 120), ReLU(),
                    Linear(120, 84), ReLU(), Linear(84, 10)
                )
                return m, CrossEntropyLoss(), (1, 28, 28), False, 'classification', [str(i) for i in range(10)]
            if arch == 'vgg11':
                m = build_vgg11(in_channels=1, num_classes=10, width=0.25)
            elif arch == 'mobilenet':
                m = build_mobilenet_v1(in_channels=1, num_classes=10, width=0.25)
            else:
                m = ResNetClassifier(in_channels=1, num_classes=10)
            return m, CrossEntropyLoss(), (1, 64, 64), True, 'classification', [str(i) for i in range(10)]
        if task == 'donkeycar':
            if arch == 'cnn':
                m = build_donkeycar_cnn()
            elif arch == 'vit':
                dim = max(16, min(int(cfg.get('vit_dim', 48)), 256))
                depth = max(1, min(int(cfg.get('vit_depth', 2)), 8))
                heads = max(1, min(int(cfg.get('vit_heads', 4)), 16))
                while dim % heads != 0 and heads > 1:
                    heads -= 1
                patch = int(cfg.get('vit_patch', 8))
                if DC_IMG_SIZE % patch != 0:
                    patch = 8
                m = build_vit(img_size=DC_IMG_SIZE, patch_size=patch, dim=dim,
                              depth=depth, heads=heads, mlp_ratio=2, in_channels=1)
            else:
                m = build_resnet(in_channels=1)
            return m, MSELoss(), (1, DC_IMG_SIZE, DC_IMG_SIZE), False, 'regression', None
        raise ValueError(f'未知任务: {task}')

    def _build_task_components(self, cfg):
        task = cfg.get('task', 'donkeycar')
        arch = cfg.get('arch', 'resnet_gn')
        seed = int(cfg.get('seed', 42))
        samples = int(cfg.get('samples', 400))
        scale_mode = cfg.get('scale', 'standard')
        augment = bool(cfg.get('augment', False))
        set_seed(seed)

        model, loss_fn, input_shape, need_upsample, task_kind, class_names = self._create_model(task, arch, cfg=cfg)
        scaler = None

        if task == 'linear':
            rng = np.random.RandomState(seed)
            n = max(40, min(samples, 600))
            X = rng.uniform(-3.0, 3.0, size=(n, 1))
            y = 3.0 * X[:, 0] + 2.0 + rng.normal(0.0, 0.35, size=n)
            X_tr, y_tr, X_va, y_va = train_test_split(X, y, test_size=0.2, seed=seed)

        elif task == 'logistic':
            n = max(60, min(samples, 800))
            X, y = make_blobs(n_samples=n, centers=2, n_features=2, seed=seed)
            X_tr, y_tr, X_va, y_va = train_test_split(X, y, test_size=0.2, stratify=y, seed=seed)
            if scale_mode == 'minmax':
                X_tr, scaler = minmax_scale(X_tr)
                X_va = apply_scale(X_va, scaler)
            elif scale_mode == 'standard':
                X_tr, scaler = standard_scale(X_tr)
                X_va = apply_scale(X_va, scaler)

        elif task == 'iris':
            X, y, names = load_iris(data_dir=DATA_DIR)
            class_names = list(names)
            X_tr, y_tr, X_va, y_va = train_test_split(X, y, test_size=0.25, stratify=y, seed=seed)
            if scale_mode == 'minmax':
                X_tr, scaler = minmax_scale(X_tr)
                X_va = apply_scale(X_va, scaler)
            elif scale_mode == 'standard':
                X_tr, scaler = standard_scale(X_tr)
                X_va = apply_scale(X_va, scaler)

        elif task == 'mnist':
            n_tr = max(100, min(samples, 8000))
            n_te = max(50, min(2000, n_tr // 3))
            X_tr, y_tr, X_va, y_va = load_mnist_arrays(
                subset_train=n_tr, subset_test=n_te, data_dir=os.path.join(_ROOT, 'data')
            )
            if augment:
                X_tr = random_shift_images(X_tr, max_shift=2, seed=seed)
            if need_upsample:
                X_tr = _upsample_64(X_tr, 64)
                X_va = _upsample_64(X_va, 64)

        elif task == 'donkeycar':
            n = max(120, min(samples, 1200))
            X, y = generate_track_dataset(n_samples=n, img_size=DC_IMG_SIZE, seed=seed)
            X_tr, y_tr, X_va, y_va = train_test_split(X, y, test_size=0.2, seed=seed)
            if augment:
                X_tr, y_tr = _augment_donkeycar(X_tr, y_tr, seed=seed)

        return {
            'task': task, 'arch': arch, 'task_kind': task_kind,
            'model': model, 'loss_fn': loss_fn, 'scaler': scaler,
            'need_upsample': need_upsample, 'class_names': class_names,
            'input_shape': input_shape,
            'X_tr': np.asarray(X_tr, dtype=np.float64),
            'y_tr': y_tr,
            'X_va': np.asarray(X_va, dtype=np.float64),
            'y_va': y_va,
        }

    # ------------------------------------------------------------------
    # 一键训练
    # ------------------------------------------------------------------
    def start_training(self, cfg):
        with self._lock:
            if self.status == 'training' and self._train_thread and self._train_thread.is_alive():
                raise RuntimeError('已有训练任务正在运行，请先停止后再启动')
            self._stop_train.clear()
            self.status = 'training'
            self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_metric': [], 'lr': []}
            self.batch_loss.clear()
            self.batch_step.clear()
            self._step = 0
            self.progress = 0.0
            self.train_started = time.time()
            self.grad_norms = []
            self.param_hists = []
            self.task_preview = {}
            self.logs.clear()

        self._train_thread = threading.Thread(
            target=self._train_worker, args=(cfg,), daemon=True
        )
        self._train_thread.start()
        return {'ok': True, 'message': '已启动后台训练'}

    def stop_training(self):
        self._stop_train.set()
        self._log('收到用户停止请求，正在安全中止本轮...')
        return {'ok': True}

    def _train_worker(self, cfg):
        try:
            task = cfg.get('task', 'donkeycar')
            arch = cfg.get('arch', 'resnet_gn')
            epochs = max(1, min(int(cfg.get('epochs', 8)), 100))
            self.epochs_total = epochs
            batch_size = max(4, min(int(cfg.get('batch_size', 32)), 256))
            lr = float(cfg.get('lr', 1e-3))
            opt_name = cfg.get('optimizer', 'adam')
            momentum = float(cfg.get('momentum', 0.9))
            clip_norm = float(cfg.get('clip_norm', 0.0))
            l2_reg = float(cfg.get('l2_reg', 0.0))
            use_es = bool(cfg.get('early_stop', False))
            use_lrs = bool(cfg.get('lr_schedule', False))
            seed = int(cfg.get('seed', 42))

            self._log(f'正在构建任务 [{task}] · 模型 [{arch}] · 数据与算子图...')
            bundle = self._build_task_components(cfg)
            model = bundle['model']
            loss_fn = bundle['loss_fn']
            task_kind = bundle['task_kind']
            X_tr, y_tr = bundle['X_tr'], bundle['y_tr']
            X_va, y_va = bundle['X_va'], bundle['y_va']

            if opt_name == 'mbgd':
                optimizer = MBGD(model.params(), lr=lr, momentum=momentum)
            else:
                optimizer = Adam(model.params(), lr=lr)

            prof = profile_model(model, bundle['input_shape'])
            with self._lock:
                self.task = task
                self.arch = arch
                self.model = model
                self.loss_fn = loss_fn
                self.scaler = bundle['scaler']
                self.need_upsample = bundle['need_upsample']
                self.class_names = bundle['class_names']
                self.input_shape = bundle['input_shape']
                if X_va.ndim >= 3:
                    self.sample_x = X_va[0:1].copy()
                    self.sample_y = y_va[0:1].copy()
                else:
                    self.sample_x = None
                    self.sample_y = None
                self.model_info = {
                    'params': int(prof['params']),
                    'flops': int(prof['flops']),
                    'latency_ms': round(float(prof['latency_ms']), 3),
                    'tag': f'{task} / {arch}',
                }
                self._refresh_stats_locked()

            self._log(
                f'构建完成：训练样本 {len(X_tr)}，验证样本 {len(X_va)}，'
                f'参数量 {prof["params"]:,}，FLOPs {prof["flops"]:,}'
            )
            self._broadcast()

            es = EarlyStopping(patience=max(2, epochs // 3), min_delta=1e-4) if use_es else None
            lrs = LRScheduler(optimizer, mode='plateau', gamma=0.5, patience=2) if use_lrs else None
            loader = DataLoader(X_tr, y_tr, batch_size=batch_size, shuffle=True, seed=seed)

            metric_name = 'MAE'
            final_metric, best_metric = None, None
            t_start = time.perf_counter()
            for epoch in range(1, epochs + 1):
                if self._stop_train.is_set():
                    break
                ep_t0 = time.perf_counter()
                model.train_mode(True)
                batch_losses = []
                last_grads = []

                n_b = max(len(loader), 1)
                for bi_, (xb, yb) in enumerate(loader):
                    if self._stop_train.is_set():
                        break
                    x_node = Node(xb)
                    pred = model(x_node)
                    loss = loss_fn(pred, yb)
                    if l2_reg > 0:
                        loss = loss + l2_penalty(model, l2_reg)
                    loss.backward()
                    if clip_norm > 0:
                        _clip_grad_norm(model.params(), clip_norm)

                    # 捕获本批每层权重梯度 L2 范数（用于监控梯度消失/爆炸）
                    last_grads = []
                    for nm, p in model.named_params():
                        if nm.endswith('.W') and p.grad is not None:
                            last_grads.append({
                                'name': nm,
                                'norm': float(np.linalg.norm(p.grad)),
                                'max': float(np.max(np.abs(p.grad))),
                            })

                    optimizer.step()
                    optimizer.zero_grad()

                    bl = float(loss.data)
                    batch_losses.append(bl)
                    self.progress = ((epoch - 1) + (bi_ + 1) / n_b) / epochs
                    with self._lock:
                        self._step += 1
                        self.batch_loss.append(round(bl, 6))
                        self.batch_step.append(self._step)

                    # 每 4 个 batch 推送一次细粒度进度，保证图像和数字任务都很丝滑
                    if len(batch_losses) % 4 == 0:
                        self._broadcast()

                if self._stop_train.is_set() and not batch_losses:
                    break

                # 验证阶段
                model.train_mode(False)
                val_preds = self._batched_predict(model, X_va, batch_size=64)
                val_loss_node = loss_fn(Node(val_preds), y_va)
                val_loss = float(val_loss_node.data)

                if task_kind == 'classification':
                    thresh = 0.0 if isinstance(loss_fn, BCEWithLogitsLoss) else 0.5
                    val_metric = accuracy(val_preds, y_va, threshold=thresh)
                    metric_label = f'Acc={val_metric*100:.2f}%'
                    metric_name = 'Acc'
                    best_metric = max(best_metric, val_metric) if best_metric is not None else val_metric
                else:
                    val_metric = float(np.mean(np.abs(val_preds.ravel() - y_va.ravel())))
                    metric_label = f'MAE={val_metric:.4f}'
                    metric_name = 'MAE'
                    best_metric = min(best_metric, val_metric) if best_metric is not None else val_metric
                final_metric = val_metric

                train_loss = float(np.mean(batch_losses)) if batch_losses else val_loss
                cur_lr = float(optimizer.lr)
                preview = self._build_task_preview(task, model, loss_fn, X_va, y_va, val_preds)

                with self._lock:
                    self.history['epoch'].append(epoch)
                    self.history['train_loss'].append(_f(train_loss))
                    self.history['val_loss'].append(_f(val_loss))
                    self.history['val_metric'].append(_f(val_metric))
                    self.history['lr'].append(_f(cur_lr))
                    self.grad_norms = last_grads[:12]
                    self.task_preview = preview
                    self._refresh_stats_locked()

                ep_s = time.perf_counter() - ep_t0
                self._log(
                    f'Epoch {epoch}/{epochs} ({ep_s:.2f}s) | '
                    f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | '
                    f'{metric_label} | lr={cur_lr:.2e}'
                )
                self._broadcast()

                if lrs is not None:
                    lrs.step(val_loss=val_loss)
                if es is not None and es(val_loss):
                    self._log(f'触发 EarlyStopping（连续 {es.patience} 轮无改善），提前结束训练。')
                    break

            total_s = time.perf_counter() - t_start
            with self._lock:
                self.status = 'stopped' if self._stop_train.is_set() else 'completed'
                if self.status == 'completed':
                    self.progress = 1.0
                prof = profile_model(model, bundle['input_shape'])
                self.model_info['latency_ms'] = round(float(prof['latency_ms']), 3)
            self._log(f'训练结束（状态: {self.status}，总耗时 {total_s:.2f}s，单帧延迟 {self.model_info["latency_ms"]} ms）')
            self._broadcast()

            # 记录本次运行到历史（供多曲线对比 / 报告导出）
            if self.history['epoch']:
                self._record_run(
                    cfg, bundle, self.status, total_s,
                    final_metric if final_metric is not None else float('nan'),
                    best_metric if best_metric is not None else float('nan'),
                    metric_name, prof, '',
                )

        except Exception as exc:
            with self._lock:
                self.status = 'error'
            self._log(f'训练异常：{exc}')
            traceback.print_exc()
            self._broadcast()

    def _batched_predict(self, model, X, batch_size=64):
        outs = []
        model.train_mode(False)
        for i in range(0, len(X), batch_size):
            outs.append(np.asarray(model(Node(X[i:i + batch_size])).data))
        return np.concatenate(outs, axis=0)

    def _build_task_preview(self, task, model, loss_fn, X_va, y_va, val_preds):
        """生成适合前端 Canvas 直接绘制的任务实时预览数据。"""
        if task == 'linear':
            xs = X_va[:, 0]
            ys = y_va.ravel()
            grid_x = np.linspace(float(xs.min()) - 0.3, float(xs.max()) + 0.3, 40).reshape(-1, 1)
            grid_y = np.asarray(model(Node(grid_x)).data).ravel()
            return {
                'kind': 'linear',
                'points': [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs[:80], ys[:80])],
                'line': [[round(float(x), 3), round(float(y), 3)] for x, y in zip(grid_x[:, 0], grid_y)],
            }
        if task == 'logistic':
            gx = np.linspace(-3.0, 3.0, 18)
            gy = np.linspace(-3.0, 3.0, 18)
            xx, yy = np.meshgrid(gx, gy)
            grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
            logits = np.asarray(model(Node(grid_pts)).data).ravel()
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            return {
                'kind': 'logistic',
                'points': [[round(float(a), 3), round(float(b), 3), int(c)]
                           for (a, b), c in zip(X_va[:120], y_va[:120])],
                'grid': np.round(probs.reshape(18, 18), 3).tolist(),
                'range': [-3.0, 3.0],
            }
        if task == 'iris':
            pred_cls = val_preds.argmax(axis=1)
            cm = np.zeros((3, 3), dtype=int)
            for t, p in zip(y_va, pred_cls):
                cm[int(t), int(p)] += 1
            return {'kind': 'iris', 'confusion': cm.tolist(), 'names': self.class_names}
        if task == 'donkeycar':
            m = regression_metrics(val_preds, y_va)
            pairs = [[round(float(t), 3), round(float(p), 3)]
                     for t, p in zip(y_va[:60].ravel(), val_preds[:60].ravel())]
            return {'kind': 'donkeycar', 'metrics': m, 'pairs': pairs}
        return {'kind': 'mnist'}

    def _refresh_stats_locked(self, max_layers=10, bins=24):
        if self.model is None:
            self.param_hists = []
            return
        items = []
        for nm, p in self.model.named_params():
            if p.data.ndim < 1:
                continue
            if p.data.ndim >= 2 or nm.endswith('.W'):
                flat = p.data.reshape(-1)
                hist, edges = np.histogram(flat, bins=bins)
                items.append({
                    'name': nm,
                    'count': int(flat.size),
                    'mean': round(float(flat.mean()), 4),
                    'std': round(float(flat.std()), 4),
                    'l2': round(float(np.sqrt(np.sum(flat ** 2))), 4),
                    'hist': [int(v) for v in hist],
                    'edges': [round(float(e), 4) for e in edges],
                })
                if len(items) >= max_layers:
                    break
        self.param_hists = items

    def snapshot(self):
        with self._lock:
            sample_info = None
            if self.sample_x is not None and self.sample_x.ndim >= 3:
                img = self.sample_x[0]
                img2d = img.mean(axis=0) if img.ndim == 3 else img
                ds = _downsample_2d(img2d, 28)
                sample_info = {
                    'shape': list(img2d.shape),
                    'pixels': np.round(ds, 3).tolist(),
                }
            return {
                'status': self.status,
                'task': self.task,
                'arch': self.arch,
                'class_names': self.class_names,
                'input_shape': list(self.input_shape),
                'model_info': dict(self.model_info),
                'history': {k: list(v) for k, v in self.history.items()},
                'batch_loss': list(self.batch_loss),
                'batch_step': list(self.batch_step),
                'epochs_done': len(self.history['epoch']),
                'epochs_total': self.epochs_total,
                'progress': round(self.progress, 4),
                'elapsed_s': round(time.time() - self.train_started, 1) if (self.train_started and self.status == 'training') else None,
                'params': list(self.param_hists),
                'grads': list(self.grad_norms),
                'logs': list(self.logs),
                'preview': dict(self.task_preview),
                'sample': sample_info,
                'last_saved_path': self.last_saved_path,
                'uptime_s': round(time.time() - self.started_at, 1),
            }

    def _broadcast(self):
        """广播训练状态快照（无 event 名，前端走 onmessage）。"""
        with self._lock:
            if not self._clients:
                return
            targets = list(self._clients)
        snap = self.snapshot()
        for q in targets:
            q.append(('', snap))

    def _broadcast_event(self, name, payload):
        """广播带名字的 SSE 事件（如 live / drive），前端用 addEventListener 收。"""
        with self._lock:
            if not self._clients:
                return
            targets = list(self._clients)
        for q in targets:
            q.append((name, payload))

    def stop(self):
        """安全停止训练与 HTTP 服务，释放端口。"""
        self._stop_train.set()
        self._stop_live.set()
        self._stop_drive.set()
        self._stop_server.set()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None

    # ------------------------------------------------------------------
    # 保存 / 加载与往返零误差校验
    # ------------------------------------------------------------------
    def save_current_model(self, tag=None):
        with self._lock:
            if self.model is None:
                raise RuntimeError('当前没有可保存的模型，请先训练或加载模型')
            name = tag or f'{self.task}_{self.arch}'
            safe = ''.join(c for c in name if c.isalnum() or c in ('_', '-')) or 'model'
            npz_path = os.path.join(STUDIO_OUT_DIR, f'{safe}.npz')
            meta_path = os.path.join(STUDIO_OUT_DIR, f'{safe}.meta.json')

            save_model(self.model, npz_path)
            meta = {
                'task': self.task,
                'arch': self.arch,
                'input_shape': list(self.input_shape),
                'need_upsample': self.need_upsample,
                'class_names': self.class_names,
                'model_info': self.model_info,
            }
            if self.scaler is not None:
                meta['scaler'] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in self.scaler.items()
                }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # 往返一致性即时校验
            probe = np.random.RandomState(0).rand(2, *self.input_shape).astype(np.float64)
            self.model.train_mode(False)
            out1 = np.asarray(self.model(Node(probe)).data)
            load_model(self.model, npz_path)
            self.model.train_mode(False)
            out2 = np.asarray(self.model(Node(probe)).data)
            max_diff = float(np.max(np.abs(out1 - out2)))
            self.last_saved_path = npz_path
        self._log(f'已保存模型到 {os.path.relpath(npz_path, _ROOT)}（往返加载最大误差 = {max_diff:.2e}）')
        return {'ok': True, 'path': os.path.relpath(npz_path, _ROOT), 'roundtrip_max_diff': max_diff}

    def load_saved_model(self, preset=None):
        """支持加载 web_studio 或 outputs/ 下预训练模型并校验往返一致性。"""
        with self._lock:
            if preset == 'donkeycar_resnet':
                npz_path = os.path.join(OUTPUTS_DIR, 'donkeycar_resnet', 'donkeycar_resnet_model.npz')
                task, arch, need_up = 'donkeycar', 'resnet_gn', False
            elif preset == 'donkeycar_cnn':
                npz_path = os.path.join(OUTPUTS_DIR, 'donkeycar', 'donkeycar_model.npz')
                task, arch, need_up = 'donkeycar', 'cnn', False
            elif preset == 'donkeycar_vit':
                npz_path = os.path.join(OUTPUTS_DIR, 'donkeycar_vit', 'donkeycar_vit_model.npz')
                task, arch, need_up = 'donkeycar', 'vit', False
            elif preset == 'mnist_lenet':
                npz_path = os.path.join(OUTPUTS_DIR, 'web', 'mnist_lenet.npz')
                task, arch, need_up = 'mnist', 'lenet', False
            elif preset == 'mnist_vgg11':
                npz_path = os.path.join(OUTPUTS_DIR, 'web', 'mnist_vgg11.npz')
                task, arch, need_up = 'mnist', 'vgg11', True
            else:
                npz_path = self.last_saved_path or os.path.join(STUDIO_OUT_DIR, f'{self.task}_{self.arch}.npz')
                meta_path = os.path.splitext(npz_path)[0] + '.meta.json'
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    task = meta['task']
                    arch = meta['arch']
                    need_up = bool(meta.get('need_upsample', False))
                    if 'scaler' in meta:
                        self.scaler = {
                            k: (np.array(v) if isinstance(v, list) else v)
                            for k, v in meta['scaler'].items()
                        }
                else:
                    task, arch, need_up = self.task, self.arch, self.need_upsample

            if not os.path.exists(npz_path):
                raise FileNotFoundError(f'未找到权重文件：{npz_path}')

            # 重建匹配骨架并加载参数
            m, loss_fn, input_shape, need_up, _, class_names = self._create_model(task, arch)

            load_model(m, npz_path)
            m.train_mode(False)
            self.task = task
            self.arch = arch
            self.model = m
            self.loss_fn = loss_fn
            self.need_upsample = need_up
            self.input_shape = input_shape
            self.class_names = class_names
            self.last_saved_path = npz_path
            prof = profile_model(m, input_shape)
            self.model_info = {
                'params': int(prof['params']),
                'flops': int(prof['flops']),
                'latency_ms': round(float(prof['latency_ms']), 3),
                'tag': f'已加载 {os.path.basename(npz_path)}',
            }
            probe = np.random.RandomState(1).rand(1, *input_shape).astype(np.float64)
            out1 = np.asarray(m(Node(probe)).data)
            load_model(m, npz_path)
            out2 = np.asarray(m(Node(probe)).data)
            diff = float(np.max(np.abs(out1 - out2)))
            self._refresh_stats_locked()

        self._log(f'成功加载模型 {os.path.relpath(npz_path, _ROOT)}（往返误差 {diff:.2e}）')
        self._broadcast()
        return {'ok': True, 'model_info': self.model_info, 'roundtrip_max_diff': diff}

    # ------------------------------------------------------------------
    # 推理、遮挡热图与 DonkeyCar 仿真
    # ------------------------------------------------------------------
    def predict(self, payload):
        with self._lock:
            if self.model is None:
                raise RuntimeError('尚未训练或加载模型')
            model = self.model
            task = self.task
            arch = self.arch
            class_names = self.class_names
            need_up = self.need_upsample
            scaler = self.scaler
            input_shape = self.input_shape

        if 'image' in payload:
            arr = np.asarray(payload['image'], dtype=np.float64)
            if arr.ndim == 2:
                arr = arr[None, None, :, :]
            elif arr.ndim == 3:
                arr = arr[None, ...]
            if need_up and arr.shape[-1] != 64:
                arr = _upsample_64(arr, 64)
        elif 'features' in payload:
            arr = np.asarray(payload['features'], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            if scaler is not None:
                arr = apply_scale(arr, scaler)
        else:
            raise ValueError('请求体缺少 image 或 features 字段')

        # 输入维度与当前模型不匹配时给出可读提示，而不是底层 unpack 报错
        # input_shape 不含 batch 维，故实际张量维度 = len(input_shape) + 1
        expected_ndim = len(input_shape) + 1
        if arr.ndim != expected_ndim or tuple(arr.shape[1:]) != tuple(input_shape):
            raise ValueError(
                f'输入形状不匹配：当前模型 [{task} / {arch}] 需要 {tuple(input_shape)} 的单样本输入'
                f'（即 {expected_ndim} 维张量 (N, {", ".join(str(d) for d in input_shape)})），'
                f'收到 {tuple(arr.shape)}。请改用与该模型匹配的任务，或先加载对应模型。'
            )

        model.train_mode(False)
        t0 = time.perf_counter()
        out = np.asarray(model(Node(arr)).data)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if task in ('mnist', 'iris'):
            probs = _softmax(np.ravel(out))
            idx = int(np.argmax(probs))
            label = class_names[idx] if (class_names and idx < len(class_names)) else str(idx)
            return {
                'task': 'classification', 'pred': idx, 'label': label,
                'probs': [round(float(v), 5) for v in probs], 'latency_ms': latency_ms
            }
        if task == 'logistic':
            logit = float(np.ravel(out)[0])
            prob1 = float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))
            idx = 1 if prob1 >= 0.5 else 0
            return {
                'task': 'classification', 'pred': idx,
                'label': class_names[idx] if class_names else str(idx),
                'probs': [round(1.0 - prob1, 5), round(prob1, 5)], 'latency_ms': latency_ms
            }
        val = float(np.ravel(out)[0])
        return {
            'task': 'regression', 'pred': val, 'value': val,
            'label': f'{val:+.3f}', 'latency_ms': latency_ms
        }

    def occlusion_heatmap(self, image=None, grid=8, patch_ratio=0.28):
        with self._lock:
            if self.model is None:
                raise RuntimeError('尚未训练或加载模型')
            model = self.model
            task = self.task
            need_up = self.need_upsample
            x = self.sample_x.copy() if image is None else np.asarray(image, dtype=np.float64)

        if x is None:
            raise RuntimeError('当前无可用图像样本')
        if x.ndim == 2:
            x = x[None, None, :, :]
        elif x.ndim == 3:
            x = x[None, ...]
        if need_up and x.shape[-1] != 64:
            x = _upsample_64(x, 64)

        _, _, H, W = x.shape
        gh, gw = H // grid, W // grid
        ph = max(2, int(gh * patch_ratio * 2.5))
        pw = max(2, int(gw * patch_ratio * 2.5))

        model.train_mode(False)
        base_out = np.asarray(model(Node(x)).data)
        if task in ('mnist',):
            probs = _softmax(np.ravel(base_out))
            top_cls = int(np.argmax(probs))
            base_score = float(probs[top_cls])
        else:
            top_cls = None
            base_score = float(np.ravel(base_out)[0])

        heat = np.zeros((grid, grid), dtype=np.float64)
        for gi in range(grid):
            for gj in range(grid):
                xm = x.copy()
                yc = gi * gh + gh // 2
                xc = gj * gw + gw // 2
                y0, y1 = max(0, yc - ph // 2), min(H, yc + (ph + 1) // 2)
                x0, x1 = max(0, xc - pw // 2), min(W, xc + (pw + 1) // 2)
                xm[:, :, y0:y1, x0:x1] = 0.0
                out = np.asarray(model(Node(xm)).data)
                if top_cls is not None:
                    sc = float(_softmax(np.ravel(out))[top_cls])
                else:
                    sc = float(np.ravel(out)[0])
                heat[gi, gj] = abs(sc - base_score)

        vmax = float(heat.max()) if heat.max() > 0 else 1.0
        return {
            'grid': np.round(heat / vmax, 4).tolist(),
            'vmax': round(vmax, 5),
            'base': round(base_score, 4),
        }

    def render_and_predict_track(self, bottom=6.0, tilt=10.0, disturb='clean'):
        """渲染 64×64 合成赛道并支持 11 种扰动预设中的代表项，实时返回预测与热图。"""
        img = render_track_view(float(bottom), float(bottom) + float(tilt), DC_IMG_SIZE) / 255.0
        X = img[None, None, :, :].astype(np.float64)
        if disturb and disturb != 'clean':
            kind = DISTURB_KINDS.get(str(disturb))
            if kind is None:
                raise ValueError(f'未知扰动类型: {disturb}')
            X = perturb_images(X, kind, np.random.RandomState(42)).astype(np.float64)
        gt = dataset_label(float(bottom), float(bottom) + float(tilt))

        res = {'image': np.round(X[0, 0], 4).tolist(), 'true_steer': round(gt, 4)}
        with self._lock:
            is_dc = (self.model is not None and self.task == 'donkeycar')
        if is_dc:
            pred_res = self.predict({'image': X})
            heat = self.occlusion_heatmap(image=X, grid=8)
            res['pred_steer'] = round(pred_res['value'], 4)
            res['latency_ms'] = pred_res['latency_ms']
            res['heatmap'] = heat['grid']
        else:
            res['pred_steer'] = None
            res['note'] = '当前激活模型非 DonkeyCar，请先训练或加载 ResNet-GN / CNN / ViT'
        return res

    def simulate_closed_loop(self, amp=12.0, wavelength=300.0, noise_sigma=0.0):
        """在正弦赛道上执行 DonkeyCar 运动学闭环仿真 vs 不转向开环对照。"""
        with self._lock:
            if self.model is None or self.task != 'donkeycar':
                raise RuntimeError('请先训练或加载 DonkeyCar 模型（如 ResNet-GN）')
            model = self.model

        model.train_mode(False)
        rows_idx = np.arange(DC_IMG_SIZE, dtype=np.float64)
        rng = np.random.RandomState(42)

        def center_fn(z):
            return float(amp) * np.sin(2.0 * np.pi * z / float(wavelength))

        def run_one(steer_enabled=True):
            x, z, theta = center_fn(0.0), 0.0, 0.0
            zs, xs, cxs, devs = [z], [x], [center_fn(z)], [0.0]
            out_of_road = False
            for _ in range(STEPS):
                if steer_enabled:
                    forward_dist = (DC_IMG_SIZE - 1 - rows_idx) * 1.2
                    z_ahead = z + forward_dist * np.cos(theta)
                    x_ahead = x + forward_dist * np.sin(theta)
                    row_offsets = np.array([center_fn(zi) for zi in z_ahead]) - x_ahead
                    view = _render_from_row_offsets(row_offsets, DC_IMG_SIZE) / 255.0
                    if noise_sigma > 0:
                        view = np.clip(view + rng.normal(0.0, float(noise_sigma), size=view.shape), 0.0, 1.0)
                    steer = float(np.ravel(model(Node(view[None, None, :, :])).data)[0])
                    theta = float(np.clip(theta + steer * DTHETA, -MAX_HEADING, MAX_HEADING))
                z += SPEED * np.cos(theta)
                x += SPEED * np.sin(theta)
                cx = center_fn(z)
                dev = abs(x - cx)
                zs.append(round(float(z), 2))
                xs.append(round(float(x), 2))
                cxs.append(round(float(cx), 2))
                devs.append(round(float(dev), 2))
                if dev > FAIL_DEV:
                    out_of_road = True
                    break
            return {
                'z': zs, 'x': xs, 'center_x': cxs,
                'max_dev': round(float(max(devs)), 2),
                'success': bool((max(devs) <= ROAD_HALF_W) and not out_of_road),
            }

        closed = run_one(steer_enabled=True)
        open_loop = run_one(steer_enabled=False)
        return {
            'closed': closed, 'open_loop': open_loop,
            'road_half_w': ROAD_HALF_W, 'arch': self.arch,
        }

    # ------------------------------------------------------------------
    # 11 项视觉扰动鲁棒性扫描
    # ------------------------------------------------------------------
    def scan_robustness(self, samples=300, seed=2026):
        """对当前 DonkeyCar 模型跑完 11 项视觉扰动，返回各扰动下的 MAE 等指标。

        协议与仓库严格评测一致：固定随机种子生成赛道数据 → 按 70/15/15 划分
        → 取测试段 → 逐项施加扰动 → 计算开环误差指标。
        """
        with self._lock:
            if self.model is None or self.task != 'donkeycar':
                raise RuntimeError('请先训练或加载 DonkeyCar 模型（如 ResNet-GN）再扫描')
            model = self.model
            arch = self.arch
            params = int(self.model_info.get('params', 0))

        n = max(60, min(int(samples), 1200))
        X, y = generate_track_dataset(n_samples=n, img_size=DC_IMG_SIZE, seed=int(seed))
        n_tr = int(round(n * TRAIN_FRAC))
        n_va = int(round(n * VAL_FRAC))
        X_te, y_te = X[n_tr + n_va:], y[n_tr + n_va:]
        if len(X_te) == 0:                      # 样本过少时兜底用全量
            X_te, y_te = X, y

        model.train_mode(False)
        rows = []
        for key, kind in DISTURB_KINDS.items():
            rng = np.random.RandomState(1000)   # 每项扰动用同一随机源，保证可比
            Xp = perturb_images(X_te, kind, rng)
            pred = predict_array(model, Xp)
            m = regression_metrics(pred, y_te)
            rows.append({
                'key': key, 'kind': kind,
                'mae': round(float(m['mae']), 5),
                'rmse': round(float(m['rmse']), 5),
                'p95_abs': round(float(m['p95_abs']), 5),
                'max_abs': round(float(m['max_abs']), 5),
                'r2': (None if np.isnan(m['r2']) else round(float(m['r2']), 4)),
            })

        base = next((r for r in rows if r['key'] == 'clean'), None)
        base_mae = base['mae'] if base else 0.0
        for r in rows:
            r['delta'] = round(r['mae'] - base_mae, 5)
            r['degrade_pct'] = (round((r['mae'] - base_mae) / base_mae * 100, 1)
                               if base_mae else None)
        worst = max(rows, key=lambda r: r['mae']) if rows else None
        result = {
            'rows': rows,
            'arch': arch, 'params': params,
            'n_test': int(len(X_te)),
            'split': [TRAIN_FRAC, VAL_FRAC, round(1 - TRAIN_FRAC - VAL_FRAC, 4)],
            'clean_mae': base_mae,
            'mean_mae': round(float(np.mean([r['mae'] for r in rows])), 5) if rows else 0.0,
            'worst': worst,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with self._lock:
            self.robustness = result
        self._log(f'鲁棒性扫描完成：{len(rows)} 项扰动，干净 MAE={base_mae:.4f}，'
                  f'最差「{worst["kind"]}」MAE={worst["mae"]:.4f}。')
        return result

    # ------------------------------------------------------------------
    # 架构自省 / 反向计算图
    # ------------------------------------------------------------------
    def architecture(self, with_weights=False, max_points=600):
        """返回当前模型的逐层结构 + 3D 布局（可选附带权重点云）。"""
        with self._lock:
            if self.model is None:
                raise RuntimeError('尚未训练或加载模型，无法展示架构')
            model = self.model
            task, arch = self.task, self.arch
            input_shape = self.input_shape

        probe = None
        if self.sample_x is not None:
            probe = np.asarray(self.sample_x[:1], dtype=np.float64)

        arch_info = trace_architecture(model, probe=probe, input_shape=tuple(input_shape))
        input_img = None
        if probe is not None and probe.ndim == 4:
            img = np.asarray(probe[0], dtype=np.float64)
            img2d = img.mean(axis=0) if img.ndim == 3 else img
            ds = _downsample_2d(img2d, 28)
            lo, hi = float(ds.min()), float(ds.max())
            rng = (hi - lo) or 1.0
            input_img = np.round((ds - lo) / rng, 3).tolist()
        out = {
            'task': task, 'arch': arch,
            'params': arch_info['params'],
            'input_shape': arch_info['input_shape'],
            'output_shape': arch_info['output_shape'],
            'layers': arch_info['layers'],
            'layout': layer_layout(arch_info['layers']),
            'model_info': dict(self.model_info),
            'input_image': input_img,
        }
        if with_weights:
            out['weights'] = weight_pointcloud(model.named_params(),
                                              max_points_per_layer=max_points)
        return out

    def backward_trace(self, sample_index=0):
        """在单样本上跑一次前向 + 反向，返回计算图结构与各节点梯度范数。"""
        with self._lock:
            if self.model is None:
                raise RuntimeError('尚未训练或加载模型，无法展示反向传播')
            model = self.model
            task, loss_fn = self.task, self.loss_fn
            sample_x, sample_y = self.sample_x, self.sample_y

        probe = None
        if sample_x is not None and len(sample_x):
            probe = np.asarray(sample_x[int(sample_index) % len(sample_x):][:1],
                               dtype=np.float64)
        if probe is None:
            probe = np.zeros((1,) + tuple(self.input_shape), dtype=np.float64)

        target = None
        if sample_y is not None and len(sample_y):
            target = np.asarray(sample_y[int(sample_index) % len(sample_y)], dtype=np.float64)
            if target.ndim == 0:
                target = target.reshape(1)
        else:
            target = np.zeros((1,), dtype=np.float64)

        if loss_fn is None:
            loss_fn = MSELoss()
        trace = trace_backward(model, loss_fn, target, probe)
        trace['task'] = task
        trace['arch'] = self.arch
        return trace

    # ------------------------------------------------------------------
    # 实时测试集评估（后台线程 + SSE 推送）
    # ------------------------------------------------------------------
    def start_live_eval(self, samples=300, seed=2026, interval=0.25):
        """启动后台线程：逐条推理当前 DonkeyCar 模型并实时推送测试集指标。"""
        with self._lock:
            if self.model is None or self.task != 'donkeycar':
                raise RuntimeError('请先训练或加载 DonkeyCar 模型再运行实时评估')
            model = self.model
            arch = self.arch
            self._stop_live.set()
        if getattr(self, '_live_thread', None) and self._live_thread.is_alive():
            self._live_thread.join(timeout=1.0)
        with self._lock:
            self._stop_live.clear()
            self._live = {'state': 'running', 'done': 0, 'total': int(samples),
                          'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'last': [], 'img': None,
                          'elapsed': 0.0, 'arch': arch, 'started_at': time.time()}

        n = max(30, min(int(samples), 1200))
        X, y = generate_track_dataset(n_samples=n, img_size=DC_IMG_SIZE, seed=int(seed))
        idx_tr = int(round(n * TRAIN_FRAC))
        idx_va = int(round(n * VAL_FRAC))
        X_te, y_te = X[idx_tr + idx_va:], y[idx_tr + idx_va:]
        if len(X_te) == 0:
            X_te, y_te = X, y

        def worker():
            model.train_mode(False)
            errs, preds, gts = [], [], []
            t0 = time.perf_counter()
            for i in range(len(X_te)):
                if self._stop_live.is_set():
                    break
                x1 = X_te[i:i + 1]
                pred = float(np.ravel(model(Node(x1)).data)[0])
                gt = float(y_te[i])
                err = abs(pred - gt)
                errs.append(err); preds.append(pred); gts.append(gt)
                # 每条都推一次（间隔由 interval 控制），保证前端看到「逐条推进」
                if i % 3 == 0 or i == len(X_te) - 1:
                    e = np.asarray(errs)
                    mae = float(e.mean())
                    rmse = float(np.sqrt((e ** 2).mean()))
                    g = np.asarray(gts); p = np.asarray(preds)
                    sst = float(((g - g.mean()) ** 2).sum())
                    r2 = float(1 - ((p - g) ** 2).sum() / sst) if sst > 0 else 0.0
                    img_2d = np.round(x1[0, 0], 4).tolist()
                    with self._lock:
                        st = self._live
                        st.update({
                            'state': 'running', 'done': len(errs), 'total': len(X_te),
                            'mae': round(mae, 5), 'rmse': round(rmse, 5), 'r2': round(r2, 5),
                            'last': [{'gt': round(gts[j], 4), 'pred': round(preds[j], 4),
                                      'err': round(errs[j], 4)}
                                     for j in range(max(0, len(errs) - 60), len(errs))],
                            'img': img_2d,
                            'elapsed': round(time.perf_counter() - t0, 2),
                            'throughput': round(len(errs) / max(1e-6, time.perf_counter() - t0), 1),
                        })
                    self._broadcast_event('live', dict(st))
                    if self._stop_live.wait(interval):
                        break
            with self._lock:
                if self._live:
                    self._live['state'] = 'stopped' if self._stop_live.is_set() else 'completed'
            self._broadcast_event('live', dict(self._live or {}))

        self._live_thread = threading.Thread(target=worker, daemon=True)
        self._live_thread.start()
        return {'ok': True, 'total': int(len(X_te))}

    def stop_live_eval(self):
        self._stop_live.set()
        return {'ok': True}

    # ------------------------------------------------------------------
    # 8 agent 同时实时驾驶（后台线程 + SSE 推送）
    # ------------------------------------------------------------------
    def start_drive(self, agents=8, amp=12.0, wavelength=300.0, noise_sigma=0.0,
                    interval=0.05, road_mode='sine', road_seed=2026,
                    policy='model', policy_every=1):
        """启动后台线程：N 台虚拟车在正弦 / 随机弯道上闭环驾驶。

        policy='model'     用当前加载的 DonkeyCar 模型推理；
        policy='heuristic' 用几何跟踪控制器（无需权重，帧率最高）；
        policy_every       每 N 步做一次模型推理，其余步复用上次转向，用于提速。
        """
        policy = 'heuristic' if str(policy).lower() in ('heuristic', 'geo', 'geometry', 'none') else 'model'
        policy_every = max(1, min(int(policy_every), 5))
        with self._lock:
            if policy == 'model':
                if self.model is None or self.task != 'donkeycar':
                    raise RuntimeError('请先加载 DonkeyCar 模型，或改用几何控制器')
                model = self.model
                arch = self.arch
            else:
                model = None
                arch = 'heuristic'
            self._stop_drive.set()
        if getattr(self, '_drive_thread', None) and self._drive_thread.is_alive():
            self._drive_thread.join(timeout=1.0)

        amp = float(amp)
        wavelength = max(60.0, float(wavelength))
        road_mode = 'random' if str(road_mode) in ('random', 'rand', '1', 'true', 'True') else 'sine'
        harmonics = [[round(amp, 4), round(wavelength, 4), 0.0]]
        if road_mode == 'random':
            rr = np.random.RandomState(int(road_seed))
            for _ in range(3):
                harmonics.append([
                    round(float(amp * rr.uniform(0.16, 0.42)), 4),
                    round(float(rr.uniform(max(75.0, wavelength * 0.32), wavelength * 0.82)), 4),
                    round(float(rr.uniform(0.0, 2.0 * math.pi)), 4),
                ])

        def center_fn(z, phase=0.0):
            v = 0.0
            for a, lam, ph in harmonics:
                v += a * math.sin(2.0 * math.pi * z / lam + ph + phase)
            return v

        with self._lock:
            self._stop_drive.clear()
            n_agents = max(1, min(int(agents), 16))
            self._drive = {
                'state': 'running', 'agents': n_agents, 'arch': arch,
                'policy': policy, 'policy_every': policy_every, 'infer_ms': 0.0,
                't': 0.0, 'step': 0, 'amp': amp,
                'wavelength': wavelength,
                'noise_sigma': float(noise_sigma),
                'road_half_w': ROAD_HALF_W,
                'road': {'mode': road_mode, 'harmonics': harmonics,
                         'half_w': ROAD_HALF_W, 'seed': int(road_seed)},
                'cars': [], 'fps': 0.0,
            }

        n_agents = self._drive['agents']
        rng = np.random.RandomState(42)
        # 前向纵队发车：#0 在最后，其余车依次在前方。
        # 这样追尾相机永远在车队尾部，不会与后车穿插，也不会在中心线叠成一团。
        cars = []
        for i in range(n_agents):
            z0 = 7.0 * i
            cars.append({
                'id': i,
                'z': z0, 'x': center_fn(z0),
                'theta': 0.0, 'dev': 0.0, 'alive': True,
                'offset': 0.0,
                'phase': 0.0,
                'seed': int(rng.randint(0, 10 ** 6)),
                'steer': 0.0,
            })

        rows_idx = np.arange(DC_IMG_SIZE, dtype=np.float64)

        def heuristic_steer(car):
            """几何跟踪：横向偏差 + 航向偏差 → 归一化转向（不依赖神经网络）。"""
            look = 32.0
            z1 = car['z'] + look * math.cos(car['theta'])
            x1 = car['x'] + look * math.sin(car['theta'])
            lat = center_fn(z1, car['phase']) - x1
            slope = (center_fn(car['z'] + look, car['phase'])
                     - center_fn(car['z'], car['phase'])) / look
            heading_err = math.atan(slope) - car['theta']
            return float(np.clip(0.055 * lat + 1.55 * heading_err, -1.0, 1.0))

        def worker():
            if model is not None:
                model.train_mode(False)
            t0 = time.perf_counter()
            step = 0
            last_infer_ms = 0.0
            while not self._stop_drive.is_set():
                step_start = time.perf_counter()
                alive_cars = [car for car in cars if car['alive']]
                infer_ms = 0.0
                if alive_cars:
                    if policy == 'heuristic':
                        for car in alive_cars:
                            car['steer'] = heuristic_steer(car)
                    elif step % policy_every == 0:
                        views = []
                        for car in alive_cars:
                            phase = car['phase']
                            forward = (DC_IMG_SIZE - 1 - rows_idx) * 1.2
                            z_ahead = car['z'] + forward * math.cos(car['theta'])
                            x_ahead = car['x'] + forward * math.sin(car['theta'])
                            offsets = np.array([center_fn(zi, phase) for zi in z_ahead]) - x_ahead
                            view = _render_from_row_offsets(offsets, DC_IMG_SIZE) / 255.0
                            if noise_sigma > 0:
                                r = np.random.RandomState(car['seed'] + step)
                                view = np.clip(view + r.normal(0.0, float(noise_sigma),
                                                               size=view.shape), 0.0, 1.0)
                            views.append(view[None, :, :])
                        t_inf = time.perf_counter()
                        batch_x = np.stack(views, axis=0)
                        steers = np.ravel(model(Node(batch_x)).data)
                        infer_ms = (time.perf_counter() - t_inf) * 1000.0
                        last_infer_ms = infer_ms
                        for car, steer_val in zip(alive_cars, steers):
                            # 低通滤波：抑制逐帧转向抖动，追尾视角更稳
                            raw = float(steer_val)
                            car['steer'] = float(0.70 * raw + 0.30 * car['steer'])

                    # 运动学每步都更新；未推理的步沿用上次转向
                    for car in alive_cars:
                        car['theta'] = float(np.clip(car['theta'] + car['steer'] * DTHETA,
                                                    -MAX_HEADING, MAX_HEADING))
                        car['z'] += SPEED * math.cos(car['theta'])
                        car['x'] += SPEED * math.sin(car['theta'])
                        cx = center_fn(car['z'], car['phase'])
                        car['dev'] = abs(car['x'] - cx)
                        car['alive'] = car['dev'] <= FAIL_DEV
                step += 1
                elapsed = time.perf_counter() - t0
                with self._lock:
                    st = self._drive
                    st.update({
                        'state': 'running', 'step': step, 't': round(elapsed, 2),
                        'fps': round(step / max(1e-6, elapsed), 1),
                        'infer_ms': round(last_infer_ms, 2),
                        'cars': [{
                            'id': c['id'], 'x': round(c['x'], 2), 'z': round(c['z'], 2),
                            'theta': round(c['theta'], 4), 'dev': round(c['dev'], 2),
                            'alive': c['alive'], 'steer': round(c['steer'], 4),
                        } for c in cars],
                    })
                self._broadcast_event('drive', dict(st))
                # 补足目标帧间隔，而不是在计算后再固定 sleep
                remain = float(interval) - (time.perf_counter() - step_start)
                if remain > 0 and self._stop_drive.wait(remain):
                    break
            with self._lock:
                if self._drive:
                    self._drive['state'] = 'stopped'
            self._broadcast_event('drive', dict(self._drive or {}))

        self._drive_thread = threading.Thread(target=worker, daemon=True)
        self._drive_thread.start()
        return {'ok': True, 'agents': n_agents, 'policy': policy,
                'policy_every': policy_every,
                'road': dict(self._drive.get('road') or {})}

    def stop_drive(self):
        self._stop_drive.set()
        return {'ok': True}

    # ------------------------------------------------------------------
    # 实验报告导出（Markdown + 图表 PNG + ZIP 打包）
    # ------------------------------------------------------------------
    def _report_charts(self):
        """用 matplotlib 渲染报告所需的 PNG 图表，返回 [(标题, 文件名, png_bytes)]。"""
        charts = []
        with self._lock:
            snap = {
                'history': {k: list(v) for k, v in self.history.items()},
                'params': list(self.param_hists),
                'grads': list(self.grad_norms),
                'model_info': dict(self.model_info),
                'task': self.task, 'arch': self.arch,
                'runs': list(self.runs),
                'robustness': self.robustness,
            }
        h = snap['history']
        if h['epoch']:
            fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
            ax.plot(h['epoch'], h['train_loss'], marker='o', ms=3.5, lw=2, label='训练损失')
            ax.plot(h['epoch'], h['val_loss'], marker='s', ms=3.5, lw=2, label='验证损失')
            ax.set_xlabel('epoch'); ax.set_ylabel('loss')
            ax.set_title('训练 / 验证损失曲线'); ax.grid(alpha=.25); ax.legend()
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)
            charts.append(('训练/验证损失曲线', '01_训练曲线.png', buf.getvalue()))

        if h['epoch'] and len(h['epoch']) == len(h['val_metric']):
            is_reg = snap['task'] in ('donkeycar', 'linear')
            fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=140)
            m = h['val_metric']
            ax.plot(h['epoch'], [v * 100 if not is_reg else v for v in m],
                    marker='o', ms=3.5, lw=2, color='#d97706')
            ax.set_xlabel('epoch')
            ax.set_ylabel('MAE' if is_reg else 'Accuracy (%)')
            ax.set_title('验证指标曲线'); ax.grid(alpha=.25)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)
            charts.append(('验证指标曲线', '02_验证指标.png', buf.getvalue()))

        if snap['params']:
            fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=140)
            names = [p['name'].replace('.W', '') for p in snap['params']]
            l2 = [p['l2'] for p in snap['params']]
            ax.barh(range(len(names)), l2, color='#030213')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel('L2 范数'); ax.set_title('各层权重 L2 范数'); ax.grid(axis='x', alpha=.25)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)
            charts.append(('各层权重 L2 范数', '03_权重范数.png', buf.getvalue()))

        if snap['grads']:
            fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=140)
            names = [g['name'].replace('.W', '') for g in snap['grads']]
            norms = [g['norm'] for g in snap['grads']]
            ax.barh(range(len(names)), norms, color='#b45309')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xscale('symlog')
            ax.set_xlabel('梯度 L2 范数'); ax.set_title('各层梯度范数（对称对数刻度）')
            ax.grid(axis='x', alpha=.25)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)
            charts.append(('各层梯度 L2 范数', '04_梯度范数.png', buf.getvalue()))

        rob = snap['robustness']
        if rob and rob.get('rows'):
            fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
            rows = rob['rows']
            ax.bar(range(len(rows)), [r['mae'] for r in rows], color='#d4183d')
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels([r['kind'] for r in rows], rotation=30, ha='right', fontsize=8)
            ax.set_ylabel('MAE'); ax.set_title('11 项视觉扰动鲁棒性（MAE，越低越好）')
            ax.grid(axis='y', alpha=.25)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)
            charts.append(('11 项扰动鲁棒性', '05_鲁棒性.png', buf.getvalue()))
        return charts

    def build_report(self):
        """生成实验报告：Markdown 正文 + 图表 PNG，返回 (markdown, charts)。"""
        with self._lock:
            snap = {
                'status': self.status, 'task': self.task, 'arch': self.arch,
                'model_info': dict(self.model_info),
                'history': {k: list(v) for k, v in self.history.items()},
                'runs': list(reversed(self.runs)),
                'robustness': self.robustness,
                'input_shape': list(self.input_shape),
                'class_names': self.class_names,
            }
        h = snap['history']
        mi = snap['model_info']
        L = []
        L.append('# MiniFlows 实验报告')
        L.append('')
        L.append(f'> 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")} ｜ '
                 f'任务：`{snap["task"]}` ｜ 架构：`{snap["arch"]}` ｜ 状态：`{snap["status"]}`')
        L.append('')

        L.append('## 一、模型概览')
        L.append('')
        L.append('| 项目 | 数值 |')
        L.append('| --- | --- |')
        L.append(f'| 参数量 | {mi.get("params", 0):,} |')
        L.append(f'| FLOPs | {mi.get("flops", 0):,}（{mi.get("flops", 0)/1e6:.2f} M） |')
        L.append(f'| 单帧推理延迟 | {mi.get("latency_ms", 0)} ms |')
        L.append(f'| 输入形状 | {tuple(snap["input_shape"])} |')
        L.append(f'| 权重来源 | {mi.get("tag", "—")} |')
        L.append('')

        if h['epoch']:
            is_reg = snap['task'] in ('donkeycar', 'linear')
            best = min(h['val_metric']) if is_reg else max(h['val_metric'])
            final = h['val_metric'][-1]
            L.append('## 二、训练结果')
            L.append('')
            L.append(f'- 训练轮数：{len(h["epoch"])} ｜ 指标口径：{"MAE（越低越好）" if is_reg else "Accuracy（越高越好）"}')
            L.append(f'- 最终验证损失：{h["val_loss"][-1]}')
            L.append(f'- 最终{"MAE" if is_reg else "准确率"}：{final if is_reg else f"{final*100:.2f}%"}')
            L.append(f'- 最优{"MAE" if is_reg else "准确率"}：{best if is_reg else f"{best*100:.2f}%"}')
            L.append('')
            L.append('| epoch | train_loss | val_loss | ' + ('MAE' if is_reg else 'accuracy') + ' | lr |')
            L.append('| ---: | ---: | ---: | ---: | ---: |')
            for i in range(len(h['epoch'])):
                L.append(f'| {h["epoch"][i]} | {h["train_loss"][i]} | {h["val_loss"][i]} | '
                         f'{h["val_metric"][i]} | {h["lr"][i]} |')
            L.append('')

        if snap['runs']:
            L.append('## 三、运行历史对比')
            L.append('')
            L.append('| # | 任务 · 架构 | 轮数 | 样本 | lr | ' +
                     ('最优 MAE' if snap['task'] in ('donkeycar', 'linear') else '最优准确率') +
                     ' | 耗时(s) | 状态 |')
            L.append('| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |')
            for r in snap['runs']:
                cfg = r.get('cfg', {})
                bm = r.get('best_metric')
                bms = ('—' if bm is None else
                       (f'{bm:.4f}' if snap['task'] in ('donkeycar', 'linear') else f'{bm*100:.2f}%'))
                L.append(f'| {r.get("id")} | {r.get("label")} | {cfg.get("epochs")} | '
                         f'{cfg.get("samples")} | {cfg.get("lr")} | {bms} | '
                         f'{r.get("elapsed_s")} | {r.get("status")} |')
            L.append('')

        rob = snap['robustness']
        if rob and rob.get('rows'):
            L.append('## 四、11 项视觉扰动鲁棒性')
            L.append('')
            L.append(f'测试集 {rob["n_test"]} 样本 ｜ 划分 '
                     f'{int(rob["split"][0]*100)}/{int(rob["split"][1]*100)}/{int(rob["split"][2]*100)} ｜ '
                     f'干净 MAE = {rob["clean_mae"]}')
            L.append('')
            L.append('| 扰动 | MAE | RMSE | P95 | 相对干净劣化 |')
            L.append('| --- | ---: | ---: | ---: | ---: |')
            for r in rob['rows']:
                deg = '—' if r.get('degrade_pct') is None else f'{r["degrade_pct"]:+.1f}%'
                L.append(f'| {r["kind"]} | {r["mae"]} | {r["rmse"]} | {r["p95_abs"]} | {deg} |')
            L.append('')

        L.append('---')
        L.append('')
        L.append('*本报告由 MiniFlows Web Studio 自动生成；'
                 '全部指标来自纯 NumPy 实现的 MiniFlows 框架实跑结果。*')
        return '\n'.join(L), self._report_charts()

    def build_report_zip(self):
        """把实验报告打包成 ZIP（Markdown + charts/*.png），返回 bytes。"""
        markdown, charts = self.build_report()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('MiniFlows_实验报告.md', markdown.encode('utf-8'))
            for title, filename, png in charts:
                zf.writestr(f'charts/{filename}', png)
        return buf.getvalue(), len(charts)


# ======================================================================
# 三、四大扩展机制实验室（一键实时运行）
# ======================================================================
def run_lab_grad_check():
    """扩展②：17 项中心差分数值梯度检验（eps=1e-5, tol=1e-5）。"""
    np.random.seed(42)
    tol, eps = 1e-5, 1e-5
    rows = []

    def rec(name, shape_str, fn, arrays):
        res = gradient_check(fn, arrays, eps=eps, tol=tol)
        rows.append({
            'name': name,
            'shape': shape_str,
            'max_rel': f'{res["max_rel_error"]:.3e}',
            'passed': bool(res['passed']),
        })

    a = np.random.randn(3, 4)
    b = np.random.randn(3, 4)
    rec('add', '(3,4) + (3,4)', lambda xs: ops.sum(ops.add(xs[0], xs[1])), [a, b])
    rec('mul', '(3,4) * (3,4)', lambda xs: ops.sum(ops.mul(xs[0], xs[1])), [a, b])

    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)
    rec('matmul', '(3,4) @ (4,5)', lambda xs: ops.sum(ops.matmul(xs[0], xs[1])), [A, B])

    ba = np.random.randn(2, 4, 3)
    bb = np.random.randn(3,)
    rec('broadcast_add', '(2,4,3) + (3,)', lambda xs: ops.sum(ops.broadcast_add(xs[0], xs[1])), [ba, bb])

    mag = np.random.uniform(0.2, 1.2, size=(4, 5))
    sgn = np.random.choice([-1.0, 1.0], size=(4, 5))
    xr = mag * sgn
    rec('relu', '(4,5)', lambda xs: ops.sum(ops.relu(xs[0])), [xr])
    rec('sigmoid', '(4,5)', lambda xs: ops.sum(ops.sigmoid(xs[0])), [np.random.randn(4, 5)])
    rec('tanh', '(4,6)', lambda xs: ops.sum(ops.tanh(xs[0])), [np.random.randn(4, 6)])

    logits = np.random.randn(5, 4)
    w_sm = np.random.randn(5, 4)
    rec('softmax', '(5,4)', lambda xs: ops.sum(ops.mul(ops.softmax(xs[0]), constant(w_sm))), [logits])

    rec('MSELoss', 'pred:(6,1), y:(6,1)', lambda xs: MSELoss()(xs[0], np.random.RandomState(1).randn(6, 1)), [np.random.randn(6, 1)])
    rec('BCEWithLogitsLoss', 'z:(8,1), t∈{0,1}', lambda xs: BCEWithLogitsLoss()(xs[0], np.array([0, 1, 1, 0, 1, 0, 1, 0])), [np.random.randn(8, 1)])
    rec('CrossEntropyLoss', 'logits:(7,4)', lambda xs: CrossEntropyLoss()(xs[0], np.array([0, 2, 1, 3, 2, 0, 1])), [np.random.randn(7, 4)])

    cx = np.random.randn(2, 3, 6, 6)
    cw = np.random.randn(4, 3, 3, 3) * 0.3
    cb = np.random.randn(4)
    rec('conv2d (im2col)', 'x:(2,3,6,6), W:(4,3,3,3)', lambda xs: ops.sum(ops.conv2d(xs[0], xs[1], xs[2], stride=1, padding=1)), [cx, cw, cb])

    # MaxPool2D 构造严格不等元素防止数值平局
    mx = np.arange(2 * 2 * 4 * 4, dtype=np.float64).reshape(2, 2, 4, 4) * 0.35
    rec('maxpool2d', 'x:(2,2,4,4), k=2, s=2', lambda xs: ops.sum(ops.maxpool2d(xs[0], 2, 2)), [mx])
    rec('avgpool2d', 'x:(2,2,4,4), k=2, s=2', lambda xs: ops.sum(ops.avgpool2d(xs[0], 2, 2)), [np.random.randn(2, 2, 4, 4)])

    tx = np.random.randn(2, 3, 4)
    wt = np.random.randn(4, 3, 2)
    rec('transpose', '(2,3,4)->(4,3,2)', lambda xs: ops.sum(ops.mul(ops.transpose(xs[0], (2, 1, 0)), constant(wt))), [tx])
    ws = np.random.randn(3, 1)
    rec('sum(axis=1)', '(3,4)->(3,1)', lambda xs: ops.sum(ops.mul(ops.sum(xs[0], axis=1, keepdims=True), constant(ws))), [np.random.randn(3, 4)])

    mlp = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    xm = np.random.randn(10, 4)
    ym = np.random.randint(0, 3, size=10)
    res_m = check_model_gradients(mlp, xm, CrossEntropyLoss(), y=ym, eps=eps, tol=tol)
    rows.append({'name': 'MLP(4→8→3)+CE 完整模型', 'shape': 'X:(10,4), 67 参数', 'max_rel': f'{res_m["max_rel_error"]:.3e}', 'passed': bool(res_m['passed'])})

    passed_n = sum(1 for r in rows if r['passed'])
    return {'rows': rows, 'total': len(rows), 'passed': passed_n, 'tol': tol, 'eps': eps}


def run_lab_graph_viz(preset='mlp'):
    """扩展①：构造真实计算图并调用 draw_computation_graph 输出 base64 PNG。"""
    np.random.seed(42)
    if preset == 'dag':
        x = Node(np.array([[1.5, -2.0]]), requires_grad=True, name='x')
        y = x * x + x
        root = ops.sum(y)
        root.op_name = 'SumLoss'
        title = '计算图: y = x*x + x 多路径共享 DAG'
    elif preset == 'resnet_block':
        blk = BasicBlock(4, 4, stride=1, num_groups=2, name='block')
        x = Node(np.random.randn(1, 4, 6, 6), requires_grad=False, name='X')
        out = blk(x)
        root = ops.sum(out)
        root.op_name = 'SumLoss'
        title = '计算图: ResNet BasicBlock（含 GroupNorm 与残差跳连）'
    else:
        mlp = Sequential(Linear(4, 16, name='fc1'), ReLU(), Linear(16, 3, name='fc2'))
        x = Node(np.random.randn(8, 4), requires_grad=False, name='X')
        root = CrossEntropyLoss()(mlp(x), np.array([0, 1, 2, 1, 0, 2, 1, 0]))
        title = '计算图: MLP(4-16-3) + CrossEntropyLoss'

    tmp_path = os.path.join(STUDIO_OUT_DIR, f'graph_{preset}.png')
    with _PLOT_LOCK:
        n_nodes, n_edges = draw_computation_graph(root, tmp_path, title=title, max_nodes=120)
        plt.close('all')

    with open(tmp_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return {
        'title': title, 'n_nodes': int(n_nodes), 'n_edges': int(n_edges),
        'image_b64': f'data:image/png;base64,{b64}',
    }


def run_lab_anomaly(kind='explosion'):
    """扩展③：在线实时跑出异常训练 vs 修复方法对照曲线。"""
    np.random.seed(42)
    if kind == 'vanishing':
        # 8 层 Sigmoid vs ReLU(He) 各层梯度范数
        X = np.random.randn(24, 16)
        y = np.random.randint(0, 3, size=24)

        def build_8(act_cls, init_mode):
            layers = []
            for i in range(7):
                layers.append(Linear(16, 16, init=init_mode, name=f'fc{i+1}'))
                layers.append(act_cls())
            layers.append(Linear(16, 3, init=init_mode, name='fc8'))
            return Sequential(*layers)

        def grad_profile(net):
            loss = CrossEntropyLoss()(net(Node(X)), y)
            loss.backward()
            vals = [round(float(np.linalg.norm(p.grad)), 6)
                    for nm, p in net.named_params() if nm.endswith('.W') and p.grad is not None]
            net.zero_grad()
            return vals

        sig_vals = grad_profile(build_8(Sigmoid, 'normal'))
        relu_vals = grad_profile(build_8(ReLU, 'he'))
        return {
            'kind': 'vanishing',
            'title': '8 层深网络各层梯度 L2 范数对比：Sigmoid+Normal vs ReLU+He',
            'labels': [f'fc{i}' for i in range(1, len(sig_vals) + 1)],
            'series': [
                {'name': 'Sigmoid + Normal（严重梯度消失）', 'values': sig_vals,
                 'color': '#1a1a1a', 'role': 'bad'},
                {'name': 'ReLU + He（梯度健康回传）', 'values': relu_vals,
                 'color': '#888888', 'role': 'good'},
            ],
        }

    if kind == 'lr_oscillation':
        rng = np.random.RandomState(42)
        X = rng.uniform(-2, 2, size=(60, 1))
        y = 3.0 * X[:, 0] + 2.0 + rng.normal(0, 0.2, size=60)
        series = []
        for lr_v, nm, col, role in [
            (1e-3, 'lr=1e-3（过慢）', '#bbbbbb', 'neutral'),
            (5e-2, 'lr=5e-2（平滑收敛）', '#888888', 'good'),
            (0.95, 'lr=0.95（震荡发散）', '#1a1a1a', 'bad'),
        ]:
            m = Sequential(Linear(1, 1))
            opt = MBGD(m.params(), lr=lr_v)
            losses = []
            for _ in range(20):
                loss = MSELoss()(m(Node(X)), y)
                loss.backward()
                opt.step()
                opt.zero_grad()
                v = float(loss.data)
                losses.append(None if (np.isnan(v) or v > 1e5) else round(v, 4))
            series.append({'name': nm, 'values': losses, 'color': col, 'role': role})
        return {
            'kind': 'lr_oscillation',
            'title': '学习率过小 / 合适 / 过大震荡对比（20 轮线性回归 MSE）',
            'labels': list(range(1, 21)),
            'series': series,
        }

    # 默认：梯度爆炸 vs 全局梯度裁剪
    X = np.random.randn(32, 16)
    y = np.random.randn(32, 1)

    def run_8(clip=False):
        np.random.seed(7)
        net = Sequential(*[
            x for i in range(7) for x in (Linear(16, 16, init='normal', init_std=0.35), ReLU())
        ], Linear(16, 1, init='normal', init_std=0.35))
        opt = Adam(net.params(), lr=0.05)
        hist = []
        for _ in range(18):
            loss = MSELoss()(net(Node(X)), y)
            loss.backward()
            if clip:
                _clip_grad_norm(net.params(), 1.0)
            opt.step()
            opt.zero_grad()
            v = float(loss.data)
            hist.append(None if (np.isnan(v) or np.isinf(v) or v > 1e6) else round(v, 4))
        return hist

    no_clip = run_8(clip=False)
    clipped = run_8(clip=True)
    return {
        'kind': 'explosion',
        'title': '8 层深网络梯度爆炸（init_std=0.35）vs 全局梯度裁剪（max_norm=1.0）',
        'labels': list(range(1, 19)),
        'series': [
            {'name': '无裁剪（急速发散）', 'values': no_clip,
             'color': '#1a1a1a', 'role': 'bad'},
            {'name': '梯度裁剪 max_norm=1.0（稳定收敛）', 'values': clipped,
             'color': '#888888', 'role': 'good'},
        ],
    }


def run_lab_preprocess():
    """扩展④：鸢尾花特征标准化前后数值对照（训练集统计量防泄露）。"""
    X, y, _ = load_iris(data_dir=DATA_DIR)
    X_tr, _, X_te, _ = train_test_split(X, y, test_size=0.25, stratify=y, seed=42)
    X_std, sc = standard_scale(X_tr)
    X_te_std = apply_scale(X_te, sc)
    feats = ['花萼长度 (cm)', '花萼宽度 (cm)', '花瓣长度 (cm)', '花瓣宽度 (cm)']
    rows = []
    for i, nm in enumerate(feats):
        rows.append({
            'feature': nm,
            'raw_mean': round(float(X_tr[:, i].mean()), 3),
            'raw_std': round(float(X_tr[:, i].std()), 3),
            'scaled_mean': round(float(X_std[:, i].mean()), 4),
            'scaled_std': round(float(X_std[:, i].std()), 4),
            'test_scaled_mean': round(float(X_te_std[:, i].mean()), 3),
        })
    return {'rows': rows, 'n_train': len(X_tr), 'n_test': len(X_te)}


def load_project_benchmarks():
    """汇总读取仓库 outputs/ 已有评测 JSON 与关键图表路径。"""
    def _read_json(rel):
        p = os.path.join(OUTPUTS_DIR, *rel.split('/'))
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    res_dc = _read_json('donkeycar_resnet/donkeycar_resnet_结果.json')
    res_vit = _read_json('donkeycar_vit/donkeycar_vit_结果.json')
    res_cnn = _read_json('donkeycar/donkeycar_测试结果.json')
    res_pt_res = _read_json('pytorch_benchmark/资源对比.json')
    res_pt_acc = _read_json('pytorch_benchmark/pytorch_result.json')
    res_tune = _read_json('tuning/调参结果.json')

    # 早期 CNN 评测 JSON 里没有写 params 字段，这里按模型定义补算，避免前端表格留空
    if isinstance(res_cnn, dict) and not res_cnn.get('params'):
        try:
            res_cnn = dict(res_cnn)
            res_cnn['params'] = count_params(build_donkeycar_cnn())
        except Exception:
            pass

    gallery = []
    candidates = [
        ('DonkeyCar 三架构综合对比（5 种子）', 'donkeycar_resnet/三方对比.png'),
        ('DonkeyCar 11 项扰动鲁棒性对比', 'donkeycar_resnet/鲁棒性对比.png'),
        ('MiniFlows vs PyTorch 单线程资源对比', 'pytorch_benchmark/资源对比.png'),
        ('ResNet-GN 多赛道闭环轨迹', 'donkeycar_resnet/多赛道闭环.png'),
        ('数据增强对精度与鲁棒性影响', 'tuning/增强对比.png'),
        ('8 层网络梯度爆炸与裁剪修复', 'exp3/梯度爆炸.png'),
    ]
    for title, rel in candidates:
        if os.path.exists(os.path.join(OUTPUTS_DIR, *rel.split('/'))):
            gallery.append({'title': title, 'url': f'/outputs/{rel}'})

    return {
        'resnet': res_dc,
        'vit': res_vit,
        'cnn': res_cnn,
        'pytorch_resources': res_pt_res,
        'pytorch_acc': res_pt_acc,
        'tuning': res_tune,
        'gallery': gallery,
    }


# ======================================================================
# 四、HTML/CSS/JS 一体化交互式工作台页面（零外部 CDN 依赖）
# ======================================================================
UI_FILE = os.path.join(_ROOT, 'applications', 'web_studio_ui.html')
ASSETS_DIR = os.path.join(_ROOT, 'applications', 'assets')


def _load_ui():
    """每次请求实时读取前端页面（改界面无需重启服务）。"""
    with open(UI_FILE, 'r', encoding='utf-8') as f:
        return f.read()



# ======================================================================
# 五、HTTP + SSE 服务端封装
# ======================================================================
def _json_default(o):
    """把 numpy 标量 / 数组转成原生 Python 类型，避免 json 序列化直接抛错。"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def start_server(host='127.0.0.1', port=8000, block=True):
    studio = MiniFlowsStudio(host=host, port=port)

    class StudioHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False, default=_json_default).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html):
            body = html.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length) if length > 0 else b'{}'
            return json.loads(raw.decode('utf-8') or '{}')

        def _send_asset(self, rel):
            """提供 applications/assets/ 下的静态资源（3D 引擎与 vendor 库）。"""
            base = os.path.abspath(ASSETS_DIR)
            target = os.path.abspath(os.path.join(base, unquote(rel).lstrip('/\\')))
            if not target.startswith(base) or not os.path.isfile(target):
                return self._send_json({'error': '文件不存在'}, code=404)
            ctype, _ = mimetypes.guess_type(target)
            if target.endswith('.js'):
                ctype = 'application/javascript; charset=utf-8'
            with open(target, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', ctype or 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(data)
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            p = parsed.path
            if p in ('/', '/index.html'):
                return self._send_html(_load_ui())
            if p == '/health':
                snap = studio.snapshot()
                return self._send_json({'ok': True, 'status': snap['status'], 'task': snap['task'], 'arch': snap['arch'], 'epochs': snap['epochs_done']})
            if p == '/api/snapshot':
                return self._send_json(studio.snapshot())
            if p == '/api/benchmarks':
                return self._send_json(load_project_benchmarks())
            if p == '/api/heatmap':
                try:
                    q = parse_qs(parsed.query)
                    grid = int(q.get('grid', ['8'])[0])
                    return self._send_json(studio.occlusion_heatmap(grid=grid))
                except Exception as exc:
                    return self._send_json({'error': str(exc)}, code=400)
            if p == '/events':
                return self._serve_sse()
            if p.startswith('/assets/'):
                return self._send_asset(p[len('/assets/'):])
            if p == '/api/runs':
                return self._send_json({'runs': studio.list_runs()})
            if p == '/api/architecture':
                q = parse_qs(parsed.query)
                with_weights = q.get('weights', ['0'])[0] in ('1', 'true', 'True')
                pts = int(q.get('points', ['600'])[0])
                return self._send_json(studio.architecture(with_weights=with_weights,
                                                           max_points=pts))
            if p == '/api/backward':
                q = parse_qs(parsed.query)
                idx = int(q.get('sample', ['0'])[0])
                return self._send_json(studio.backward_trace(sample_index=idx))
            if p == '/api/report.md':
                markdown, _ = studio.build_report()
                body = markdown.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/markdown; charset=utf-8')
                self.send_header('Content-Disposition',
                                 'attachment; filename="MiniFlows_report.md"')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if p == '/api/report.zip':
                data, n_charts = studio.build_report_zip()
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition',
                                 'attachment; filename="MiniFlows_report.zip"')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('X-Chart-Count', str(n_charts))
                self.end_headers()
                self.wfile.write(data)
                return
            if p.startswith('/outputs/'):
                rel = unquote(p[len('/outputs/'):]).lstrip('/\\')
                target = os.path.abspath(os.path.join(OUTPUTS_DIR, rel))
                if not target.startswith(os.path.abspath(OUTPUTS_DIR)) or not os.path.isfile(target):
                    return self._send_json({'error': '文件不存在'}, code=404)
                ctype, _ = mimetypes.guess_type(target)
                with open(target, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', ctype or 'application/octet-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return self._send_json({'error': 'not found'}, code=404)

        def do_POST(self):
            p = urlparse(self.path).path
            try:
                body = self._read_json()
                if p == '/api/train/start':
                    return self._send_json(studio.start_training(body))
                if p == '/api/train/stop':
                    return self._send_json(studio.stop_training())
                if p == '/api/model/save':
                    return self._send_json(studio.save_current_model(body.get('tag')))
                if p == '/api/model/load':
                    return self._send_json(studio.load_saved_model(body.get('preset')))
                if p == '/predict':
                    return self._send_json(studio.predict(body))
                if p == '/api/heatmap':
                    return self._send_json(studio.occlusion_heatmap(image=body.get('image'), grid=int(body.get('grid', 8))))
                if p == '/api/donkeycar/render':
                    return self._send_json(studio.render_and_predict_track(
                        bottom=body.get('bottom', 6.0), tilt=body.get('tilt', 10.0), disturb=body.get('disturb', 'clean')
                    ))
                if p == '/api/donkeycar/sim':
                    return self._send_json(studio.simulate_closed_loop(
                        amp=body.get('amp', 12.0), wavelength=body.get('wavelength', 300.0), noise_sigma=body.get('noise_sigma', 0.0)
                    ))
                if p == '/api/donkeycar/robustness':
                    return self._send_json(studio.scan_robustness(
                        samples=body.get('samples', 300), seed=body.get('seed', 2026)
                    ))
                if p == '/api/runs/clear':
                    return self._send_json(studio.clear_runs())
                if p == '/api/runs/delete':
                    return self._send_json(studio.delete_run(body.get('id')))
                if p == '/api/live/eval':
                    return self._send_json(studio.start_live_eval(
                        samples=body.get('samples', 300),
                        seed=body.get('seed', 2026),
                        interval=float(body.get('interval', 0.25)),
                    ))
                if p == '/api/live/stop':
                    return self._send_json(studio.stop_live_eval())
                if p == '/api/drive/start':
                    return self._send_json(studio.start_drive(
                        agents=body.get('agents', 8),
                        amp=body.get('amp', 12.0),
                        wavelength=body.get('wavelength', 300.0),
                        noise_sigma=body.get('noise_sigma', 0.0),
                        interval=float(body.get('interval', 0.05)),
                        road_mode=body.get('road_mode', 'sine'),
                        road_seed=body.get('road_seed', 2026),
                        policy=body.get('policy', 'model'),
                        policy_every=body.get('policy_every', 1),
                    ))
                if p == '/api/drive/stop':
                    return self._send_json(studio.stop_drive())
                if p == '/api/lab/grad_check':
                    return self._send_json(run_lab_grad_check())
                if p == '/api/lab/graph_viz':
                    return self._send_json(run_lab_graph_viz(body.get('preset', 'mlp')))
                if p == '/api/lab/anomaly':
                    return self._send_json(run_lab_anomaly(body.get('kind', 'explosion')))
                if p == '/api/lab/preprocess':
                    return self._send_json(run_lab_preprocess())
                return self._send_json({'error': 'not found'}, code=404)
            except Exception as exc:
                return self._send_json({'error': str(exc)}, code=400)

        def _serve_sse(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            q = deque(maxlen=24)
            with studio._lock:
                studio._clients.append(q)
            try:
                first = f'data: {json.dumps(studio.snapshot(), ensure_ascii=False, default=_json_default)}\n\n'
                self.wfile.write(first.encode('utf-8'))
                self.wfile.flush()
                while not studio._stop_server.is_set():
                    # 8ms 轮询：驾驶遥测必须逐条即时送达；120ms 批量发送会造成一顿一顿
                    time.sleep(0.008)
                    wrote = False
                    while q:
                        name, payload = q.popleft()
                        body = json.dumps(payload, ensure_ascii=False, default=_json_default)
                        msg = (f'event: {name}\n' if name else '') + f'data: {body}\n\n'
                        self.wfile.write(msg.encode('utf-8'))
                        wrote = True
                    if wrote:
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with studio._lock:
                    if q in studio._clients:
                        studio._clients.remove(q)

    class StudioServer(ThreadingHTTPServer):
        """浏览器刷新 / 断开 SSE 时会抛连接重置，属于正常现象，不刷屏。"""

        daemon_threads = True
        allow_reuse_address = True

        def handle_error(self, request, client_address):
            exc = sys.exc_info()[1]
            if isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
                return
            super().handle_error(request, client_address)

    server = StudioServer((host, port), StudioHandler)
    server.daemon_threads = True
    studio._server = server
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    studio._http_thread = t
    url = f'http://{host}:{port}/'
    print(f'[MiniFlows Studio] Web 平台已启动：{url} （按 Ctrl+C 停止）')
    if block:
        try:
            while not studio._stop_server.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print('\n[MiniFlows Studio] 正在关闭服务...')
        finally:
            studio._stop_server.set()
            studio._stop_train.set()
            server.shutdown()
            server.server_close()
    return studio, url


def main():
    parser = argparse.ArgumentParser(description='MiniFlows 全功能交互式 Web 平台')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    start_server(host=args.host, port=args.port, block=True)


if __name__ == '__main__':
    main()
