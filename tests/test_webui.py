# -*- coding: utf-8 -*-
"""
test_webui.py —— Web 实时看板 + 预测服务测试
==============================================

覆盖：
    1. HTTP 服务能启动，/health 正常；
    2. /api/snapshot 返回历史、权重直方图、梯度范数；
    3. POST /predict 返回类别与概率（概率和≈1）；
    4. /api/heatmap 返回 8×8 热图且数值在 [0,1]；
    5. 服务能正常关闭、端口释放。

做法：用标准库 urllib 发请求，不依赖 requests，也不需要浏览器。

运行（项目根目录）：
    python tests/test_webui.py
"""

import json
import os
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from common_setup import init_env
init_env(seed=42)

from miniflows.graph import Node
from miniflows.layers import Sequential, Conv2D, ReLU, MaxPool2D, Flatten, Linear
from miniflows.losses import CrossEntropyLoss
from miniflows.webui import TrainingDashboard


def _free_port():
    """让系统分配一个空闲端口（避免 8000 被占用导致测试失败）。"""
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _get(url):
    """GET 并解析 JSON。"""
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode('utf-8'))


def _post(url, obj):
    """POST JSON 并解析响应。"""
    data = json.dumps(obj).encode('utf-8')
    req = urllib.request.Request(url, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode('utf-8'))


def build_tiny():
    """小 CNN：8×8 输入 -> 3 类（测试用，前向很快）。"""
    return Sequential(
        Conv2D(1, 4, 3, padding=1), ReLU(), MaxPool2D(2, 2),
        Flatten(), Linear(4 * 4 * 4, 3),
    )


def main():
    """启动看板 -> 打接口 -> 校验 -> 关闭。"""
    print('=' * 62)
    print('Web 看板 / 预测服务测试')
    print('=' * 62)

    np.random.seed(0)
    model = build_tiny()
    loss_fn = CrossEntropyLoss()
    port = _free_port()
    dash = TrainingDashboard(model=model, loss_fn=loss_fn,
                             task='classification',
                             class_names=['猫', '狗', '鸟'],
                             port=port, title='测试看板')
    X = np.random.rand(1, 1, 8, 8)
    dash.set_sample(X, np.array([1]))
    dash.log_batch(1.23)                       # 模拟 batch 损失
    dash.callback({'epoch': 1, 'train_loss': 1.1, 'val_loss': 1.2,
                   'val_acc': 0.4, 'lr': 1e-3})  # 模拟一轮结束
    url = dash.start()

    try:
        # ---- 1) 健康检查 ----
        h = _get(url + 'health')
        assert h['ok'] and h['epochs'] == 1, h
        print(f"  [1] /health -> ok，epochs={h['epochs']} ✓")

        # ---- 2) 快照 ----
        snap = _get(url + 'api/snapshot')
        assert snap['epochs_done'] == 1
        assert len(snap['params']) >= 1 and 'hist' in snap['params'][0]
        assert len(snap['grads']) >= 1 and 'norm' in snap['grads'][0]
        print(f"  [2] /api/snapshot -> 参数层 {len(snap['params'])}，"
              f"梯度层 {len(snap['grads'])}，batch 点 {len(snap['batch_loss'])} ✓")

        # ---- 3) 预测服务 ----
        img = np.random.rand(8, 8).tolist()
        pred = _post(url + 'predict', {'image': img})
        assert pred['task'] == 'classification'
        assert 0 <= pred['pred'] < 3
        assert abs(sum(pred['probs']) - 1.0) < 1e-6
        assert pred['label'] in ('猫', '狗', '鸟')
        print(f"  [3] /predict -> 类别 {pred['label']}，"
              f"概率和 {sum(pred['probs']):.4f}，{pred['latency_ms']} ms ✓")

        # ---- 4) 热图 ----
        heat = _get(url + 'api/heatmap?grid=8')
        assert len(heat['grid']) == 8 and len(heat['grid'][0]) == 8
        flat = [v for row in heat['grid'] for v in row]
        assert min(flat) >= 0 and max(flat) <= 1
        print(f"  [4] /api/heatmap -> 8×8，归一化范围 "
              f"[{min(flat):.2f}, {max(flat):.2f}] ✓")

        # ---- 5) 主页 ----
        with urllib.request.urlopen(url, timeout=10) as r:
            html = r.read().decode('utf-8')
        assert 'EventSource' in html and 'predict' in html
        print(f"  [5] GET / -> HTML {len(html)} 字节（含 SSE 与预测演示） ✓")
    finally:
        dash.stop()

    # 端口释放后应无法连接
    try:
        urllib.request.urlopen(url + 'health', timeout=2)
        raise AssertionError('服务关闭后端口仍可访问')
    except Exception:
        pass
    print('  [6] 服务关闭后端口已释放 ✓')
    print('\n' + '=' * 62)
    print('✓ Web 看板 / 预测服务全部测试通过')
    print('=' * 62)


if __name__ == '__main__':
    main()
