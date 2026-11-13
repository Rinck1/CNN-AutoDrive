# `miniflows/webui.py` 阅读指南

> 前置：Trainer、模型参数/梯度、HTTP 基础 ｜ 难度：★★★ ｜ 建议 80 分钟

## 1. 文件定位

课程要求“web 端实时可视化训练界面 + 模型部署”。本模块用**纯标准库**实现：

- `TrainingDashboard`：训练时实时推送指标，并提供权重/梯度/热图；
- HTTP 服务：`/`（页面）、`/events`（SSE）、`/api/snapshot`、`/api/heatmap`、
  `/predict`（部署接口）、`/health`。

不依赖 Flask/FastAPI，也不需要前端构建。

## 2. 结构速览

| 顺序 | 符号 | 作用 |
|---|---|---|
| 1 | `TrainingDashboard.__init__` | 保存模型/损失/任务/端口，初始化历史与锁 |
| 2 | `log_batch` | 记录 batch 损失 |
| 3 | `callback` | Trainer 回调：epoch 指标入 history 并广播 |
| 4 | `set_sample` | 设置热图/预测演示的样本 |
| 5 | `_param_stats` | 逐层权重直方图 + μ/σ/L2 |
| 6 | `_grad_stats` | 按需 forward+backward，取逐层梯度范数（缓存 2s） |
| 7 | `snapshot` | 汇总成 JSON 可序列化的快照 |
| 8 | `occlusion_heatmap` | 8×8 遮挡敏感性 |
| 9 | `predict` | 预测服务核心（分类/回归两种输出） |
| 10 | `start` / `stop` / `serve_forever` | HTTP 生命周期 |
| 11 | `DASHBOARD_HTML` | 自包含前端（Canvas 绘图 + EventSource） |

## 3. 逐段讲解

### 3.1 与 Trainer 的接口

```python
dash = TrainingDashboard(model, loss_fn, task='classification',
                         class_names=[str(i) for i in range(10)])
url = dash.start()                     # 先启动，训练时就能看
trainer.train(..., callbacks=[dash.callback])
```

`callback(state)` 读取 `epoch/train_loss/val_loss/val_acc/lr` 写入 `history`，
然后 `_broadcast()` 给所有 SSE 客户端，返回 `False`（不早停）。

batch 级损失需要包一层损失：

```python
class LoggedLoss:
    def __call__(self, pred, target):
        loss = self.loss_fn(pred, target)
        self.dashboard.log_batch(float(loss.data))
        return loss
```

这样不改 `Trainer` 就能拿到每一步的损失。

### 3.2 SSE 推送

- 浏览器 `EventSource('/events')` 建立长连接；
- 服务端为每个客户端维护 `deque(maxlen=8)`；
- 连接时先推一次全量快照，之后每次指标更新推一次；
- 客户端断开（BrokenPipe/ConnectionReset）静默移除。

为什么不用轮询：轮询会反复触发快照计算（含梯度重算），SSE 只在有更新时推。

### 3.3 权重分布

只取名字以 `.W` 结尾的参数（跳过 bias/γ/β），做 30-bin 直方图。
返回 `count/mean/std/l2/hist/edges`，前端用柱状图画。

### 3.4 梯度范数

训练每步 `zero_grad()`，梯度不保留。所以快照时：

1. 取保存的样本 `sample_x/sample_y`；
2. `model.zero_grad()` → forward → loss → `backward()`；
3. 逐层取 `‖p.grad‖₂`；
4. `finally: model.zero_grad()` 清掉，避免污染训练。

结果缓存 2 秒，防止浏览器刷新把训练拖慢。

### 3.5 遮挡热图（因果式解释）

```python
for gi in range(grid):
    for gj in range(grid):
        xm = x.copy(); xm[..., 该块] = 0
        heat[gi,gj] = |score(xm) - score(原图)|
```

- 分类：`score` = 真实类概率；
- 回归：`score` = 转向值。

为什么要它：注意力权重不是解释。项目实测 ViT 的 [CLS] 注意力归一化熵 0.79/0.90、
与转向相关≈0；注意力换成均匀权重后 MAE 从 0.045 崩到 0.34——机制有用，
但热图不可解释。遮挡是“去掉它看变化”，更接近因果。

### 3.6 预测服务

```python
POST /predict  {"image": [[...], ...]}
-> {"task":"classification","pred":7,"label":"7","probs":[...],"latency_ms":0.4}
```

形状自动识别：`(H,W)` → `(1,1,H,W)`；`(C,H,W)` → `(1,C,H,W)`。
分类走数值稳定 softmax，回归直接返回标量。

### 3.7 线程安全

- `_lock` 保护 history/客户端列表/样本；
- `ThreadingHTTPServer` + `daemon_threads`，每个请求一个线程；
- 服务只监听 `127.0.0.1`，无鉴权，仅适合本地演示。

### 3.8 前端要点

- 纯 Canvas 绘图，无 CDN 依赖；
- 损失图：batch 细线 + epoch 圆点，**x 轴按 batch 位置对齐**（否则 2 个 epoch
  点只占最左一小段），纵轴用对数；
- 权重直方图：下拉框选层；
- 热图按钮触发 `/api/heatmap`，用 `rgb` 手工映射蓝→黄→红。

## 4. 调用关系

`applications/mnist_web.py` 构建模型 → 启动看板 → 训练 → 常驻服务；
`tests/test_webui.py` 用 urllib 打全部接口。

## 5. 易错点

1. **OMP 冲突**：torchvision 的 MKL 与 numpy 的 MKL 在 HTTP 线程里可能触发
   `OMP: Error #15`。解决：`MINIFLOWS_NO_TORCHVISION=1`（mnist_web 已默认设置）；
2. **梯度是 None**：不在快照里重新 backward，梯度图会是空的；
3. **SSE 缓冲**：必须 `flush()`，否则浏览器收不到；
4. **端口占用**：`start()` 前用空闲端口，测试里用 `_free_port()`；
5. **训练结束后**：`serve_forever()` 阻塞主线程，`--no-serve` 用于 CI；
6. **不要对外网开放**：无鉴权、无速率限制。

## 6. 自检问题

1. SSE 与轮询相比省了什么？慢客户端怎么处理？
2. 为什么梯度要按需重算而不是每步记录？
3. 遮挡热图与注意力图的本质区别？
4. `/predict` 如何判断输入形状？
5. 训练线程与 HTTP 线程共享哪些状态？锁保护了什么？
