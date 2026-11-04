# `applications/donkeycar.py` 阅读指南

> 前置：CNN 回归、matplotlib 像素渲染 ｜ 难度：★★ ｜ 建议 45 分钟

## 1. 文件定位

DonkeyCar 自动驾驶的**演示版**：用 numpy 直接渲染合成赛道图，训练 CNN 回归转向角，
并做简化运动学的闭环仿真。严格评测在 `donkeycar_eval.py`，ViT 版在 `donkeycar_vit.py`。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `_render_from_row_offsets` (87) | 按每行道路中心偏移渲染灰度图 |
| 2 | `render_track_view` (113) | 底部/顶部偏移 → 倾斜赛道图 |
| 3 | `dataset_label` (134) | 几何偏移 → 转向角标签 |
| 4 | `generate_track_dataset` (156) | 批量生成 (X,y) |
| 5 | `build_model` (183) | CNN 回归模型（38,369 参数） |
| 6 | `closed_loop_demo` (212) | 闭环/开环轨迹仿真 |
| 7 | `main` (290) | 数据/训练/评估/出图/保存/闭环 |

## 3. 逐段讲解

### 3.1 图像渲染

`_render_from_row_offsets`：

- 第 r 行道路中心列 = `img_size/2 + offset[r]`；
- 每个像素到中心列的距离 `d`；
- `d <= 10` → 路面（灰度 130）；
- `|d-10| <= 0.9` → 亮边线（235）；
- 其余背景（35）。

全部用 numpy 广播，无逐像素循环，生成一张 64×64 图极快。

### 3.2 标签公式

```python
raw = K_TILT*(top-bottom) + K_OFFSET*bottom
label = raw / LABEL_SCALE          # 归一化到 [-1,1]
```

- 第一项“跟弯”：车头对准道路延伸方向；
- 第二项“纠偏”：把车拉回中线；
- 正=右转，负=左转，与真车 PWM 约定一致。

### 3.3 数据集

`generate_track_dataset(600)`：

- 底部偏移 ∈[-18,18]，倾斜 ∈[-20,20]；
- 每张图 `(1,64,64)`，归一化到 [0,1]；
- 标签 ∈[-0.96,0.96]。

### 3.4 模型

```
Conv2D(1→8,k5,s2) → ReLU → MaxPool2D(2)   → (N,8,15,15)
Conv2D(8→16,k3)   → ReLU → MaxPool2D(2)   → (N,16,6,6)
Flatten → Linear(576→64) → ReLU → Linear(64→1) → Tanh
```

参数量 38,369；Tanh 保证输出 ∈[-1,1]。

### 3.5 训练与指标

- 600 训练 / 120 测试；
- 8 轮，Adam lr=1e-3，batch=32；
- 若 MAE≥0.1 自动加训到 15 轮；
- 演示版测试 MAE 0.0446、MSE 0.00283。

注意：`trainer.train(..., X_val=X_test)` 把测试集当验证集（`donkeycar.py:315`），
指标偏乐观；严格指标见 `donkeycar_eval.py`（MAE 0.0534±0.0069）。

### 3.6 闭环仿真

- 赛道中心线 `cx(z)=12*sin(2πz/300)`，半宽 10px；
- 每步前进 2.5px，共 100 步；
- 每步按位姿渲染 64×64 视图 → 模型预测 δ → 航向 `θ += δ*0.05`；
- 对照：不转向开环；
- 结果：闭环最大偏差 6.24px，开环 12.00px（压出路面）。

## 4. 调用关系

`render/label → generate_dataset → build_model → Trainer →
predict → closed_loop_demo → save_model`；
`donkeycar_eval.py` 复用 `generate_track_dataset/build_model/_render_from_row_offsets`。

## 5. 易错点

1. **测试集当验证集**：演示指标偏乐观；
2. **标签方向**：offset 正=道路在右=右转=正标签，别搞反；
3. **闭环不是“偷看标签”**：控制输入只能用模型预测，不能读真实 y；
4. **模型 npz 不入库**：`.gitignore` 排除了 `*.npz`；
5. **合成数据局限**：不能外推到真车。

## 6. 自检问题

1. 画出“偏移→图像→标签”的几何关系；
2. 为什么 Tanh 适合转向角？
3. 闭环/开环对照说明了什么？
4. 演示版与严格评测版的协议差异？
