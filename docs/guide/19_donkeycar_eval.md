# `applications/donkeycar_eval.py` 阅读指南

> 前置：`donkeycar.py`、Trainer、闭环仿真 ｜ 难度：★★★ ｜ 建议 90 分钟 ｜ 评测重点

## 1. 文件定位

DonkeyCar 的**严格评测版**：解决演示脚本“测试集兼作验证集、只跑一次、只跑一条赛道”的问题。

协议：

- 900 条合成图，70%/15%/15% 训练/验证/测试；
- 测试集全程不参与训练、早停、模型选择；
- 5 个随机种子，12 轮，Adam lr=1e-3，batch=32；
- 开环 + 11 种扰动 + 多赛道闭环 + 噪声闭环 + 持久化/延迟/边界。

## 2. 结构速览

| 区块 | 行号 | 内容 |
|---|---|---|
| 超参 | 76~91 | seeds/samples/epochs/speed/steps |
| 划分/指标 | 95~151 | `split_indices`、`regression_metrics`、`predict_array` |
| 训练 | 155~204 | `train_one_seed` |
| 鲁棒性 | 208~267 | `box_blur`、`perturb_images` |
| 闭环 | 271~391 | `make_predict_fn`、`closed_loop_run`、`sample_tracks` |
| 绘图 | 395~510 | 训练/误差/鲁棒性/闭环/延迟 |
| 主流程 | 514~740 | 逐种子训练、汇总、JSON |
| 报告 | 741+ | `write_report` |

## 3. 逐段讲解

### 3.1 划分

```python
perm = RandomState(seed).permutation(n)
train 70% / val 15% / test 15%
```

每个种子对应不同划分，用于评估“划分敏感性”；测试集只在训练结束后评估一次。

### 3.2 指标

`regression_metrics` 返回 MSE/RMSE/MAE/P95/max/R²。
P95 用来暴露长尾错误，R² 用来判断模型是否真的解释了方差。
直行基线（预测 0）也要算，用来证明模型比“什么都不做”好。

### 3.3 训练

`train_one_seed`：

- 固定 `np.random.seed(seed)` 初始化；
- 固定 12 轮，**不做早停/按验证集选模型**；
- 训练后 `predict_array` 在测试集评估。

### 3.4 鲁棒性

`perturb_images` 支持 11 种：

- 高斯噪声 σ=0.02/0.05/0.10；
- 亮度 ±0.2、对比度 ×0.7/×1.3；
- 3×3 均值模糊；
- 中心 16×16 遮挡、底部 16 行遮挡。

每种扰动在 5 个种子上算 MAE，报告 mean±std 和相对干净集变化。

### 3.5 闭环

`closed_loop_run`：

```
cx(z) = A·sin(2πz/λ + φ)
θ ← clip(θ + δ·0.05, ±0.6)
x += v·sinθ; z += v·cosθ
dev = x - cx(z)
```

- 每步按当前位姿渲染 64×64 视图；
- 模型输出 δ；开环时 δ=0；
- 成功判据：全程 |dev| ≤ 10px（路面半宽）；
- 偏差超过 25px 提前终止（彻底失控）。

`sample_tracks` 三类：

- 同分布：A∈[8,12]，λ∈[280,380]；
- 分布外：A∈[14,16]，λ∈[220,250]；
- 极端：A∈[18,22]，λ∈[160,190]，速度 3.0。

### 3.6 结果（5 种子）

| 测试 | 结果 |
|---|---|
| 开环 MAE | **0.0534±0.0069**，R² 0.973 |
| 直行基线 MAE | 0.3575 |
| 同分布闭环 | 100%，最大偏差 2.90px |
| 分布外闭环 | 100%，5.75px |
| 极端闭环 | **0%**，20.69px |
| 噪声闭环 σ=0.05 | 100% |
| 不转向开环 | 0% |
| 保存/加载 | max_diff=0 |
| 单帧推理 | 约 0.5ms |

### 3.7 产物

`outputs/donkeycar/`：

- `测试报告.md`；
- `donkeycar_测试结果.json`；
- 5 张图：训练曲线、开环误差分布、鲁棒性、多赛道闭环、推理延迟。

## 4. 调用关系

复用 `donkeycar.generate_track_dataset/build_model/_render_from_row_offsets`；
`donkeycar_vit.py` 复用本文件的 `split_indices/regression_metrics/perturb_images/
closed_loop_run/sample_tracks/make_predict_fn`，保证协议一致。

## 5. 易错点

1. **测试集不能用于早停/选模型**：本脚本刻意固定轮数；
2. **闭环成功判据**：是全程最大偏差 ≤10px，不是最终偏差；
3. **极端赛道失败是预期结果**：它用于找边界，不是 bug；
4. **噪声闭环的 rng**：每个模型固定种子，保证可复现；
5. **报告数字要与 JSON 一致**：JSON 是机器可读的权威来源。

## 6. 自检问题

1. 为什么 70/15/15 比 80/20 更严谨？
2. P95 与 MAE 各暴露什么问题？
3. 闭环成功率与开环 MAE 为什么不是一回事？
4. 极端赛道 0% 说明什么？下一步该怎么改？
