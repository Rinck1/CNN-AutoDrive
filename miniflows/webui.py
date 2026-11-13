# -*- coding: utf-8 -*-
"""
miniflows/webui.py —— Web 端实时训练可视化看板 + 预测服务
============================================================

课程要求：
    3) 完成 web 端实时可视化训练界面，实时展示关键训练指标（损失、准确率），
       并通过图表展示权重分布、梯度变化、图像热图等；
    1) 框架提供“模型预测、模型部署、可视化”服务。

本模块用 **Python 标准库**（http.server + SSE）实现，不依赖 Flask/FastAPI，
也不需要 Node/前端构建，`python xxx.py` 一条命令即可打开看板。

能力
----
    TrainingDashboard.start()      启动 HTTP 服务（默认 127.0.0.1:8000）
    dash.callback                 Trainer 回调：每轮把指标推给浏览器
    dash.log_batch(loss)          记录 batch 级损失（曲线更细）
    GET  /                       实时看板页面（自包含 HTML/JS，无外部资源）
    GET  /events                 Server-Sent Events：指标/权重/梯度快照推送
    GET  /api/snapshot           当前快照 JSON（便于脚本/其他前端消费）
    GET  /api/heatmap            遮挡敏感性热图（模型“看哪里”）
    POST /predict                预测服务：JSON 图像 -> 类别/转向角（部署接口）
    GET  /health                 健康检查

实现要点
--------
1. **SSE 而不是轮询**：浏览器 `EventSource` 订阅 `/events`，服务端有更新就推；
2. **梯度按需计算**：Trainer 每步 `zero_grad()`，梯度不保留；快照请求时
   对一小批样本重新 forward+backward，取每层 W 的 L2 范数后立刻清零；
3. **权重分布**：直接读参数 `data` 做直方图（30 bins），按层返回；
4. **热图**：对样本做 8×8 遮挡（occlusion），观察预测概率/转向值的变化，
   得到“影响最大的区域”，比注意力图更接近因果解释；
5. **线程安全**：更新与读取都走同一把锁，快照是纯 Python 结构，容易被 JSON 化；
6. **只监听 127.0.0.1**：本地演示服务，不对外网暴露。

依赖：numpy + 标准库；不引入任何第三方 Web 框架。
"""

import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from miniflows.graph import Node


# ======================================================================
# 一、看板状态与统计
# ======================================================================
class TrainingDashboard:
    """实时训练看板 + 预测服务。

    参数：
        model: 已构建的 MiniFlows 模型（用于权重/梯度/热图/预测）；
        loss_fn: 损失函数（用于按需计算梯度）；
        task (str): 'classification' 或 'regression'；
        class_names (list[str]|None): 分类任务的类别名；
        host/port: 监听地址（默认仅本机）；
        title (str): 页面标题；
        max_batches (int): batch 损失曲线保留的点数（防止内存增长）。
    """

    def __init__(self, model=None, loss_fn=None, task='classification',
                 class_names=None, host='127.0.0.1', port=8000,
                 title='MiniFlows 实时训练看板', max_batches=300):
        self.model = model
        self.loss_fn = loss_fn
        assert task in ('classification', 'regression')
        self.task = task
        self.class_names = list(class_names) if class_names else None
        self.host = host
        self.port = int(port)
        self.title = title

        # ---- 训练历史（epoch 级）----
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': [],
                        'val_acc': [], 'lr': []}
        # ---- batch 级损失（最近 max_batches 个）----
        self.batch_loss = deque(maxlen=max_batches)
        self.batch_step = deque(maxlen=max_batches)
        self._step = 0

        # ---- 展示用样本（热图/预测演示）----
        self.sample_x = None            # (1,C,H,W)
        self.sample_y = None
        self.last_pred = None

        # ---- 梯度缓存（按需重算，避免每步都算）----
        self._grad_cache = {'time': 0.0, 'layers': []}
        self._lock = threading.RLock()
        self._clients = []              # SSE 客户端队列
        self._server = None
        self._thread = None
        self._stop_flag = threading.Event()
        self.started_at = time.time()

    # ------------------------------------------------------------------
    # 训练接口
    # ------------------------------------------------------------------
    def log_batch(self, loss):
        """记录一个 batch 的损失（训练曲线更细）。"""
        with self._lock:
            self._step += 1
            self.batch_loss.append(float(loss))
            self.batch_step.append(self._step)

    def callback(self, state):
        """Trainer 回调：每个 epoch 结束调用一次，记录指标并推送。

        返回 False（不触发早停）；如需早停请在外层再包一层回调。
        """
        with self._lock:
            self.history['epoch'].append(int(state.get('epoch', 0)))
            self.history['train_loss'].append(_f(state.get('train_loss')))
            self.history['val_loss'].append(_f(state.get('val_loss')))
            self.history['val_acc'].append(_f(state.get('val_acc')))
            self.history['lr'].append(_f(state.get('lr')))
        self._broadcast('update')
        return False

    def set_sample(self, x, y=None):
        """设置用于热图/预测演示的样本（x: (1,C,H,W) 或 (C,H,W)）。"""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            x = x[None, ...]
        with self._lock:
            self.sample_x = x
            self.sample_y = None if y is None else np.asarray(y)

    # ------------------------------------------------------------------
    # 统计快照
    # ------------------------------------------------------------------
    def _param_stats(self, max_layers=10, bins=30):
        """权重分布：每层做直方图 + 均值/标准差/L2。"""
        if self.model is None:
            return []
        items = []
        for name, p in self.model.named_params():
            if p.data.ndim < 2 or not name.endswith('.W'):
                continue                          # 只看权重矩阵，忽略 bias/γ/β
            flat = p.data.reshape(-1)
            if flat.size == 0:
                continue
            hist, edges = np.histogram(flat, bins=bins)
            items.append({
                'name': name,
                'count': int(flat.size),
                'mean': float(flat.mean()),
                'std': float(flat.std()),
                'l2': float(np.sqrt(np.sum(flat ** 2))),
                'hist': [int(v) for v in hist],
                'edges': [float(e) for e in edges],
            })
            if len(items) >= max_layers:
                break
        return items

    def _grad_stats(self, min_interval=2.0):
        """梯度范数：对样本重新 forward+backward，取每层 W 的 L2 范数。

        缓存 min_interval 秒，避免浏览器刷新就把训练拖慢。
        """
        if self.model is None or self.loss_fn is None or self.sample_x is None:
            return []
        now = time.time()
        with self._lock:
            if now - self._grad_cache['time'] < min_interval:
                return self._grad_cache['layers']
        layers = []
        try:
            self.model.train_mode(False)
            self.model.zero_grad()
            pred = self.model(Node(self.sample_x.copy()))
            if self.sample_y is None:
                # 没给标签时用“输出平方和”作为标量目标，只用于看梯度量级
                loss = _sum_square(pred)
            else:
                loss = self.loss_fn(pred, self.sample_y)
            loss.backward()
            for name, p in self.model.named_params():
                if name.endswith('.W') and p.grad is not None:
                    layers.append({'name': name,
                                   'norm': float(np.linalg.norm(p.grad)),
                                   'max': float(np.max(np.abs(p.grad)))})
        except Exception as exc:                   # 统计失败不应中断训练
            layers = [{'name': f'<计算失败: {exc}>', 'norm': 0.0, 'max': 0.0}]
        finally:
            self.model.zero_grad()
        with self._lock:
            self._grad_cache = {'time': now, 'layers': layers}
        return layers

    def snapshot(self):
        """完整快照（JSON 可序列化），供 /api/snapshot 与 SSE 使用。"""
        with self._lock:
            data = {
                'title': self.title,
                'task': self.task,
                'class_names': self.class_names,
                'history': {k: list(v) for k, v in self.history.items()},
                'batch_loss': list(self.batch_loss),
                'batch_step': list(self.batch_step),
                'epochs_done': len(self.history['epoch']),
                'uptime_s': time.time() - self.started_at,
                'sample': None,
            }
            if self.sample_x is not None:
                img = self.sample_x[0]
                img2d = img.mean(axis=0) if img.ndim == 3 else img
                data['sample'] = {
                    'shape': list(img2d.shape),
                    'pixels': _downsample(img2d, 16).round(3).tolist(),
                }
        data['params'] = self._param_stats()
        data['grads'] = self._grad_stats()
        return data

    # ------------------------------------------------------------------
    # 热图（遮挡敏感性）
    # ------------------------------------------------------------------
    def occlusion_heatmap(self, grid=8, patch_ratio=0.25):
        """遮挡敏感性热图：逐块遮挡图像，观察预测变化。

        分类任务：记录“真实类别的概率”下降幅度；
        回归任务：记录输出变化绝对值。
        返回 grid×grid 的矩阵（已归一化到 0~1）与原始值范围。

        为什么不用注意力图：注意力权重不是因果解释（项目实测 ViT 的
        [CLS] 注意力接近均匀、与转向相关≈0）；遮挡是因果式的“去掉它看变化”。
        """
        if self.model is None or self.sample_x is None:
            raise RuntimeError('未设置模型或样本')
        x = self.sample_x.astype(np.float64).copy()
        _, C, H, W = x.shape
        gh, gw = H // grid, W // grid
        ph = max(1, int(gh * patch_ratio))
        pw = max(1, int(gw * patch_ratio))

        self.model.train_mode(False)
        base = self._predict_arrays(x)            # 基线输出
        base_score = self._score_for_heatmap(base)

        heat = np.zeros((grid, grid))
        for gi in range(grid):
            for gj in range(grid):
                xm = x.copy()
                y0, x0 = gi * gh + gh // 2 - ph // 2, gj * gw + gw // 2 - pw // 2
                xm[:, :, y0:y0 + ph, x0:x0 + pw] = 0.0   # 遮挡（置黑）
                out = self._predict_arrays(xm)
                heat[gi, gj] = abs(self._score_for_heatmap(out) - base_score)
        vmax = float(heat.max()) if heat.max() > 0 else 1.0
        return {'grid': (heat / vmax).round(4).tolist(),
                'raw': heat.round(6).tolist(),
                'vmax': vmax, 'base': float(base_score)}

    def _predict_arrays(self, x):
        """内部：对 numpy 图像前向，返回 numpy 输出。"""
        self.model.train_mode(False)
        return np.asarray(self.model(Node(x)).data)

    def _score_for_heatmap(self, out):
        """把模型输出映射成一个标量，便于比较遮挡前后变化。"""
        if self.task == 'regression':
            return float(np.ravel(out)[0])
        logits = np.ravel(out)
        probs = _softmax(logits)
        if self.sample_y is not None:
            return float(probs[int(np.ravel(self.sample_y)[0])])
        return float(probs.max())

    # ------------------------------------------------------------------
    # 预测服务
    # ------------------------------------------------------------------
    def predict(self, image):
        """预测接口：输入 (C,H,W) 或 (H,W) 的 numpy/嵌套列表，返回结果字典。

        返回：
            {'task', 'pred', 'label', 'probs'(分类), 'value'(回归), 'latency_ms'}
        """
        if self.model is None:
            raise RuntimeError('未加载模型')
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f'图像应为 (H,W)/(C,H,W)/(1,C,H,W)，得到 {arr.shape}')
        t0 = time.perf_counter()
        out = self._predict_arrays(arr)
        latency = (time.perf_counter() - t0) * 1000
        if self.task == 'classification':
            logits = np.ravel(out)
            probs = _softmax(logits)
            idx = int(np.argmax(probs))
            label = (self.class_names[idx] if self.class_names
                     and idx < len(self.class_names) else str(idx))
            res = {'task': 'classification', 'pred': idx, 'label': label,
                   'probs': [float(v) for v in probs]}
        else:
            val = float(np.ravel(out)[0])
            res = {'task': 'regression', 'pred': val, 'label': f'{val:+.3f}',
                   'value': val}
        res['latency_ms'] = round(latency, 3)
        self.last_pred = res
        return res

    # ------------------------------------------------------------------
    # HTTP 服务
    # ------------------------------------------------------------------
    def start(self, block=False):
        """启动后台 HTTP 线程，返回访问地址。"""
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            """极简路由：/、/events、/api/*、/predict、/health。"""

            def log_message(self, fmt, *args):
                """静默访问日志（避免刷屏），需要调试时改成 print。"""
                return

            def _send_json(self, obj, code=200):
                """发送 JSON 响应。"""
                body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)

            def _send_text(self, text, ctype='text/html; charset=utf-8', code=200):
                """发送文本响应（HTML/纯文本）。"""
                body = text.encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                """处理 GET：页面、SSE、快照、热图、健康检查。"""
                path = urlparse(self.path)
                if path.path in ('/', '/index.html'):
                    return self._send_text(DASHBOARD_HTML.replace(
                        '__TITLE__', dashboard.title))
                if path.path == '/health':
                    return self._send_json({'ok': True,
                                            'epochs': len(dashboard.history['epoch'])})
                if path.path == '/api/snapshot':
                    return self._send_json(dashboard.snapshot())
                if path.path == '/api/heatmap':
                    try:
                        q = parse_qs(path.query)
                        grid = int(q.get('grid', ['8'])[0])
                        return self._send_json(dashboard.occlusion_heatmap(grid=grid))
                    except Exception as exc:
                        return self._send_json({'error': str(exc)}, code=400)
                if path.path == '/events':
                    return self._serve_sse()
                return self._send_json({'error': 'not found'}, code=404)

            def do_POST(self):
                """处理 POST：/predict 预测服务。"""
                if urlparse(self.path).path != '/predict':
                    return self._send_json({'error': 'not found'}, code=404)
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    payload = json.loads(self.rfile.read(length) or b'{}')
                    image = payload.get('image')
                    if image is None:
                        return self._send_json({'error': '缺少 image 字段'}, code=400)
                    return self._send_json(dashboard.predict(image))
                except Exception as exc:
                    return self._send_json({'error': str(exc)}, code=400)

            def _serve_sse(self):
                """SSE：先推一次快照，之后指标更新时推送。"""
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                q = deque(maxlen=8)
                dashboard._clients.append(q)
                try:
                    self._sse_write(q, dashboard.snapshot())     # 首帧：全量数据
                    while not dashboard._stop_flag.is_set():
                        time.sleep(0.8)
                        if q:
                            self._sse_write(q, q.popleft())
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    if q in dashboard._clients:
                        dashboard._clients.remove(q)

            def _sse_write(self, q, data):
                """写一条 SSE 事件；客户端断开时静默。"""
                body = f'data: {json.dumps(data, ensure_ascii=False)}\n\n'
                self.wfile.write(body.encode('utf-8'))
                self.wfile.flush()

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        url = f'http://{self.host}:{self.port}/'
        print(f'[看板] 已启动：{url}（Ctrl+C 结束）')
        if block:
            self.serve_forever()
        return url

    def _broadcast(self, msg='update'):
        """向所有 SSE 客户端投递一条消息（队列满就丢最旧的）。"""
        if not self._clients:
            return
        snap = self.snapshot()
        for q in list(self._clients):
            q.append(snap)

    def serve_forever(self):
        """阻塞主线程（训练结束后继续提供预测服务）。"""
        try:
            while not self._stop_flag.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print('\n[看板] 收到 Ctrl+C，关闭服务')
        finally:
            self.stop()

    def stop(self):
        """关闭 HTTP 服务与后台线程。"""
        self._stop_flag.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        print('[看板] 服务已停止')


# ======================================================================
# 二、小工具
# ======================================================================
def _f(v):
    """None -> NaN（JSON 里用 null 表示），其余转 float。"""
    return float('nan') if v is None else float(v)


def _softmax(z):
    """数值稳定的 softmax（用于预测接口返回概率）。"""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def _downsample(img, size):
    """把二维图像降采样到 size×size（均值池化），用于看板缩略显示。"""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    bh, bw = max(1, h // size), max(1, w // size)
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            block = img[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            out[i, j] = block.mean() if block.size else 0.0
    return out


def _sum_square(node):
    """没有标签时的标量化目标：Σ out²（仅用于观察梯度量级）。"""
    from miniflows import ops
    return ops.sum(ops.mul(node, node))


# ======================================================================
# 三、前端页面（自包含：无 CDN、无构建、无外部资源）
# ======================================================================
DASHBOARD_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark;
    --bg:#0f1115; --panel:#171a21; --line:#262b36; --fg:#e6e9ef;
    --muted:#8b93a7; --acc:#4c9aff; --ok:#35c07a; --warn:#f2a93b; --bad:#ef5f5f; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f6f7f9; --panel:#ffffff; --line:#e3e6ec; --fg:#1c2230;
      --muted:#5d667a; --acc:#2b6cb0; --ok:#1f8a55; --warn:#b4711a; --bad:#c03434; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font:14px/1.5 -apple-system,"Microsoft YaHei",Segoe UI,sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; font-weight:600; }
  .pill { font-size:12px; color:var(--muted); border:1px solid var(--line);
    border-radius:999px; padding:2px 10px; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--ok);
    display:inline-block; margin-right:6px; }
  main { padding:16px 20px 40px; display:grid; gap:16px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line);
    border-radius:10px; padding:12px 14px; }
  .card .k { font-size:12px; color:var(--muted); }
  .card .v { font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:900px) { .grid2 { grid-template-columns:1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line);
    border-radius:10px; padding:12px 14px; }
  .panel h2 { font-size:13px; margin:0 0 8px; color:var(--muted); font-weight:500; }
  canvas { width:100%; display:block; }
  select, button, input { font:inherit; color:var(--fg); background:var(--bg);
    border:1px solid var(--line); border-radius:6px; padding:4px 8px; }
  button { cursor:pointer; }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .legend { font-size:12px; color:var(--muted); display:flex; gap:12px; }
  .sw { width:10px; height:10px; border-radius:2px; display:inline-block; margin-right:4px; }
  #heat { image-rendering: pixelated; border-radius:6px; }
  .pred { font-size:13px; color:var(--muted); min-height:20px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="pill"><span class="dot" id="dot"></span><span id="status">等待数据…</span></span>
  <span class="pill">epoch <b id="ep">0</b></span>
  <span class="pill">运行 <b id="up">0</b>s</span>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="k">训练损失</div><div class="v" id="k_loss">–</div></div>
    <div class="card"><div class="k">验证损失</div><div class="v" id="k_vloss">–</div></div>
    <div class="card"><div class="k">验证指标</div><div class="v" id="k_acc">–</div></div>
    <div class="card"><div class="k">学习率</div><div class="v" id="k_lr">–</div></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>损失曲线（细线=batch 损失，圆点=epoch 训练/验证，对数纵轴）</h2>
      <canvas id="c_loss" height="200"></canvas>
      <div class="legend"><span><i class="sw" style="background:var(--acc)"></i>train</span>
        <span><i class="sw" style="background:var(--warn)"></i>val</span></div>
    </div>
    <div class="panel">
      <h2>验证指标 / 学习率</h2>
      <canvas id="c_acc" height="200"></canvas>
      <div class="legend"><span><i class="sw" style="background:var(--ok)"></i>指标</span>
        <span><i class="sw" style="background:var(--muted)"></i>lr</span></div>
    </div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>权重分布</h2>
      <div class="row" style="margin-bottom:8px">
        <select id="layer"></select>
        <span class="pill" id="wstat">–</span>
      </div>
      <canvas id="c_hist" height="190"></canvas>
    </div>
    <div class="panel">
      <h2>梯度范数（每层 W，按需重算）</h2>
      <div class="row" style="margin-bottom:8px">
        <button id="refresh_grad">重新计算梯度</button>
        <span class="pill" id="gmax">–</span>
      </div>
      <canvas id="c_grad" height="190"></canvas>
    </div>
  </div>

  <div class="panel">
    <h2>遮挡敏感性热图（模型“去掉这块最影响输出”）</h2>
    <div class="row">
      <button id="heat_btn">计算热图</button>
      <span class="pill" id="heat_info">未计算</span>
    </div>
    <div class="row" style="margin-top:10px; align-items:flex-start">
      <div style="flex:0 0 auto"><canvas id="c_input" width="128" height="128"></canvas>
        <div class="pred" style="text-align:center">输入样本</div></div>
      <div style="flex:0 0 auto"><canvas id="heat" width="128" height="128"></canvas>
        <div class="pred" style="text-align:center">遮挡热图</div></div>
      <div class="pred" id="pred_box" style="flex:1 1 240px"></div>
    </div>
  </div>
</main>
<script>
const S = { snap: null };
const el = id => document.getElementById(id);
const F = (v, d=4) => (v==null || Number.isNaN(v)) ? '–' : Number(v).toFixed(d);

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function axes(ctx, w, h, pad) {
  ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h-pad);
  ctx.lineTo(w-pad, h-pad); ctx.stroke();
}

// 折线图：每个 series 自带 x 数组（batch 与 epoch 采样频率不同，必须各用各的横轴）
function lineChart(id, seriesList) {
  const c = el(id); const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = Math.round(rect.width*dpr); c.height = 200*dpr;
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height, pad = 38*dpr;
  ctx.clearRect(0,0,w,h); axes(ctx, w, h, pad);
  const all = seriesList.flatMap(s => s.y).filter(v => !Number.isNaN(v));
  if (!all.length) return;
  const xs = seriesList.flatMap(s => s.x).filter(v => v!=null && !Number.isNaN(v));
  const xmax = Math.max(...xs, 1);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (lo === hi) { lo -= 1; hi += 1; }
  // 对数纵轴：损失跨 2.88 -> 0.29 两个量级，线性轴会把后期压成一条线
  const useLog = lo > 0;
  const Y = v => {
    if (!useLog) return h-pad - (h-2*pad)*(v-lo)/(hi-lo);
    const a = Math.log10(lo), b = Math.log10(hi);
    return h-pad - (h-2*pad)*(Math.log10(v)-a)/((b-a)||1);
  };
  const X = v => pad + (w-2*pad)*v/xmax;
  ctx.font = `${11*dpr}px sans-serif`; ctx.fillStyle = css('--muted');
  ctx.fillText(useLog ? hi.toPrecision(3) : hi.toExponential(2), 4*dpr, pad+4*dpr);
  ctx.fillText(useLog ? lo.toPrecision(3) : lo.toExponential(2), 4*dpr, h-pad);
  seriesList.forEach(s => {
    ctx.strokeStyle = css(s.color); ctx.lineWidth = (s.width||2)*dpr;
    ctx.beginPath(); let started=false;
    s.y.forEach((v,i) => {
      const xv = s.x[i];
      if (Number.isNaN(v) || xv==null || Number.isNaN(xv)) return;
      if (!started) { ctx.moveTo(X(xv), Y(v)); started=true; } else ctx.lineTo(X(xv), Y(v));
    });
    ctx.stroke();
    if (s.dots) {                          // epoch 点加圆点，2 个点也能看清
      s.y.forEach((v,i) => {
        const xv = s.x[i];
        if (Number.isNaN(v) || xv==null) return;
        ctx.fillStyle = css(s.color);
        ctx.beginPath(); ctx.arc(X(xv), Y(v), 3.2*dpr, 0, Math.PI*2); ctx.fill();
      });
    }
  });
}

function histChart(id, hist, edges) {
  const c = el(id); const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = Math.round(rect.width*dpr); c.height = 190*dpr;
  const ctx = c.getContext('2d'); const w = c.width, h = c.height, pad = 30*dpr;
  ctx.clearRect(0,0,w,h); axes(ctx,w,h,pad);
  if (!hist || !hist.length) return;
  const m = Math.max(...hist, 1);
  const bw = (w-2*pad)/hist.length;
  hist.forEach((v,i) => {
    const bh = (h-2*pad)*v/m;
    ctx.fillStyle = css('--acc');
    ctx.fillRect(pad+i*bw, h-pad-bh, Math.max(1,bw-1), bh);
  });
  ctx.font = `${10*dpr}px sans-serif`; ctx.fillStyle = css('--muted');
  ctx.fillText(Number(edges[0]).toFixed(3), pad, h-6*dpr);
  const last = Number(edges[edges.length-1]).toFixed(3);
  ctx.fillText(last, w-pad-ctx.measureText(last).width, h-6*dpr);
}

function barChart(id, names, values) {
  const c = el(id); const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = Math.round(rect.width*dpr); c.height = 190*dpr;
  const ctx = c.getContext('2d'); const w = c.width, h = c.height, pad = 30*dpr;
  ctx.clearRect(0,0,w,h); axes(ctx,w,h,pad);
  if (!values || !values.length) return;
  const m = Math.max(...values, 1e-12);
  const bw = (w-2*pad)/values.length;
  values.forEach((v,i) => {
    const bh = (h-2*pad)*v/m;
    ctx.fillStyle = css('--warn');
    ctx.fillRect(pad+i*bw+bw*0.15, h-pad-bh, bw*0.7, bh);
  });
  ctx.font = `${10*dpr}px sans-serif`; ctx.fillStyle = css('--muted');
  ctx.fillText('max '+m.toExponential(2), pad, 12*dpr);
}

function grayCanvas(id, pixels) {
  const c = el(id); const ctx = c.getContext('2d');
  const g = pixels.length, cell = c.width/g;
  ctx.clearRect(0,0,c.width,c.height);
  let lo = 1e9, hi = -1e9;
  pixels.flat().forEach(v => { lo=Math.min(lo,v); hi=Math.max(hi,v); });
  const span = (hi-lo) || 1;
  for (let i=0;i<g;i++) for (let j=0;j<g;j++) {
    const v = Math.round(255*(pixels[i][j]-lo)/span);
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.fillRect(j*cell, i*cell, Math.ceil(cell), Math.ceil(cell));
  }
}

function colorCanvas(id, grid) {
  const c = el(id); const ctx = c.getContext('2d');
  const g = grid.length, cell = c.width/g;
  ctx.clearRect(0,0,c.width,c.height);
  for (let i=0;i<g;i++) for (let j=0;j<g;j++) {
    const t = Math.max(0, Math.min(1, grid[i][j]));
    // 蓝 -> 黄 -> 红 的简易热力色
    const r = Math.round(255*Math.min(1, t*1.8));
    const gg = Math.round(255*Math.max(0, Math.min(1, t*1.8-0.5)));
    const b = Math.round(255*Math.max(0, 0.55-t*1.6));
    ctx.fillStyle = `rgb(${r},${gg},${b})`;
    ctx.fillRect(j*cell, i*cell, Math.ceil(cell), Math.ceil(cell));
  }
}

function render(snap) {
  S.snap = snap;
  const H = snap.history;
  el('status').textContent = snap.epochs_done ? '训练中/已连接' : '已连接（等待首个 epoch）';
  el('ep').textContent = snap.epochs_done;
  el('up').textContent = Math.round(snap.uptime_s);
  const last = a => a.length ? a[a.length-1] : null;
  el('k_loss').textContent = F(last(H.train_loss));
  el('k_vloss').textContent = F(last(H.val_loss));
  el('k_acc').textContent = F(last(H.val_acc), snap.task==='classification'?4:4);
  el('k_lr').textContent = last(H.lr)==null ? '–' : Number(last(H.lr)).toExponential(2);

  const bl = snap.batch_loss.slice();
  const ne = H.train_loss.length;
  // epoch 指标映射到 batch 横轴（1..nb），这样两条曲线共享同一时间轴
  const nb = Math.max(bl.length, 2);
  const ex = ne ? H.train_loss.map((_,i)=> (nb-1)*(i+0.5)/ne) : [];
  const ax = H.train_loss.map((_,i)=>i);
  lineChart('c_loss', [
    {x: bl.map((_,i)=>i), y: bl, color:'--acc', width:1},
    {x: ex, y: H.train_loss, color:'--acc', width:2.5, dots:true},
    {x: ex, y: H.val_loss, color:'--warn', width:2.5, dots:true},
  ]);
  lineChart('c_acc', [
    {x: ax, y: H.val_acc, color:'--ok', width:2.5, dots:true},
    {x: ax, y: H.lr, color:'--muted', width:1.5},
  ]);

  const sel = el('layer'); const params = snap.params || [];
  const prev = sel.value;
  if (params.length && sel.options.length !== params.length) {
    sel.innerHTML = params.map((p,i)=>`<option value="${i}">${p.name}</option>`).join('');
    if (prev) sel.value = prev;
  }
  if (params.length) {
    const p = params[Number(sel.value)||0];
    el('wstat').textContent = `${p.count} 个 · μ=${F(p.mean,3)} σ=${F(p.std,3)} ‖W‖=${F(p.l2,2)}`;
    histChart('c_hist', p.hist, p.edges);
  } else { el('wstat').textContent = '暂无参数'; }

  const grads = snap.grads || [];
  barChart('c_grad', grads.map(g=>g.name), grads.map(g=>g.norm));
  if (grads.length) el('gmax').textContent = `最大梯度 ${grads.reduce((a,b)=>a.norm>b.norm?a:b).norm.toExponential(2)}`;

  if (snap.sample) grayCanvas('c_input', snap.sample.pixels);
  if (snap.prediction) el('pred_box').textContent = '预测：' + JSON.stringify(snap.prediction);
}

el('layer').addEventListener('change', () => S.snap && render(S.snap));
el('refresh_grad').addEventListener('click', async () => {
  const r = await fetch('/api/snapshot'); render(await r.json());
});
el('heat_btn').addEventListener('click', async () => {
  el('heat_info').textContent = '计算中…';
  const r = await fetch('/api/heatmap?grid=8');
  const d = await r.json();
  if (d.error) { el('heat_info').textContent = '失败：' + d.error; return; }
  colorCanvas('heat', d.grid);
  el('heat_info').textContent = `8×8 遮挡 · 最大影响 ${Number(d.vmax).toExponential(2)}`;
});

const es = new EventSource('/events');
es.onmessage = ev => { try { render(JSON.parse(ev.data)); } catch (e) { console.error(e); } };
es.onerror = () => { el('status').textContent = '连接中断，重试中…'; };
</script>
</body>
</html>'''
