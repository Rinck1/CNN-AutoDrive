# `applications/mnist_web.py` 阅读指南

> 前置：`webui.py`、`classic_cnn.py`、`Trainer` ｜ 难度：★★ ｜ 建议 30 分钟

## 1. 文件定位

把“训练 + Web 看板 + 预测服务”串起来的演示脚本：

```bash
python applications/mnist_web.py                          # LeNet + 看板
python applications/mnist_web.py --model vgg11 --width 0.5
python applications/mnist_web.py --model mobilenet --width 0.25
python applications/mnist_web.py --load                   # 只部署
python applications/mnist_web.py --epochs 1 --no-serve     # CI/自检
```

## 2. 结构速览

| 顺序 | 符号 | 作用 |
|---|---|---|
| 1 | `build_lenet` | 28×28 → 10 类，44,426 参数 |
| 2 | `build_model_by_name` | 返回 (model, 是否需要上采样) |
| 3 | `upsample` | 28×28 → 64×64 最近邻（给 VGG/MobileNet/ResNet） |
| 4 | `LoggedLoss` | 损失包装器：把 batch 损失喂给看板 |
| 5 | `main` | 数据 → 看板 → 训练/加载 → 演示预测 → 常驻 |

## 3. 逐段讲解

### 3.1 为什么先启动看板

```python
dash = TrainingDashboard(model=model, loss_fn=CrossEntropyLoss(), ...)
dash.set_sample(X_te[:1], y_te[:1])
url = dash.start()          # 训练之前就起服务
```

先起服务，训练第一轮就能在浏览器看到曲线；`set_sample` 让热图/梯度有样本可用。

### 3.2 batch 损失的注入

`Trainer` 只回调 epoch，不暴露 batch。这里用 `LoggedLoss` 包一层：

```python
loss = self.loss_fn(pred, target)
self.dashboard.log_batch(float(loss.data))
return loss
```

不改变计算图，梯度照常回传。

### 3.3 模型切换

- LeNet 直接吃 28×28；
- VGG/MobileNet/ResNet 需要更大分辨率，`upsample` 用最近邻放大到 64×64
  （索引取样，无插值开销）；
- `input_shape` 随之变化，`profile_model` 用它算 FLOPs/延迟。

### 3.4 训练 vs 部署

- 默认：训练 → `save_model` 存 `outputs/web/mnist_<model>.npz` → 常驻服务；
- `--load`：跳过训练，加载 npz 后直接提供 `/predict`；
- `--no-serve`：训练完调用 `dash.stop()` 退出，适合自动化。

### 3.5 实测（2500 样本 3 轮 LeNet）

```
[Trainer] 轮次 1/3 | 0.?? | val_acc 0.6?
[Trainer] 轮次 3/3 | 训练损失 0.3801 | 验证损失 0.4953 | 验证准确率 0.8327
模型规模：44,426 参数，0.57 MFLOPs，单帧 0.56 ms，激活 0.08 MB
```

## 4. 产物

- `outputs/web/mnist_<model>.npz`：参数；
- `outputs/web/训练摘要.json`：最终指标 + 参数量/FLOPs/延迟。

## 5. 易错点

1. **必须先设 `MINIFLOWS_NO_TORCHVISION=1`**（脚本已自动设）：
   torchvision 的 MKL 与 numpy 的 MKL 在 HTTP 线程里冲突会触发 OMP #15；
2. `--load` 前必须存在对应 npz，否则退出并提示；
3. VGG/MobileNet 在 600 样本 1 轮下准确率只有 0.23，属正常欠拟合，
   不要当成模型错误；
4. 端口被占用时先改 `--port`；
5. 常驻服务会阻塞，演示结束用 Ctrl+C。

## 6. 自检问题

1. 为什么要先启动看板再训练？
2. `LoggedLoss` 会不会影响梯度？
3. 三种模型分别需要什么输入尺寸？
4. 如何在另一台机器上只做部署？
