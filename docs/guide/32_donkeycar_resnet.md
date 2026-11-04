# `applications/donkeycar_resnet.py` 阅读指南

> 前置：`resnet.py`、`donkeycar_eval.py` ｜ 难度：★★★ ｜ 建议 60 分钟

## 1. 文件定位

DonkeyCar 的 ResNet 版本，与 CNN / ViT 使用**完全相同的协议**：

- 900 样本、70/15/15、5 种子、12 轮、Adam lr=1e-3、batch=32；
- 测试集不参与训练/早停/选模型；
- 开环 + 11 种扰动 + 多赛道闭环 + 噪声闭环 + 保存/加载 + 延迟 + 边界输入。

## 2. 结构速览

| 区块 | 行号 | 内容 |
|---|---|---|
| 训练 | `train_one_seed` | 同协议训练 ResNet |
| 鲁棒性 | `robustness_suite` | 11 种扰动 |
| 绘图 | `plot_*` | 训练/误差/三方对比/鲁棒性/闭环/延迟 |
| 主流程 | `main` | 训练、闭环、持久化、JSON |
| 报告 | `write_report` | 中文报告 |

## 3. 逐段讲解

### 3.1 模型

```python
build_resnet()   # 44,337 参数
```

- Stem：Conv(1→8,k5,s2,p2) → GroupNorm → ReLU；
- 3 个 stage，每个 2 个 BasicBlock，通道 8/16/32；
- 全局平均池化 → Linear(32→1) → Tanh；
- 用 GroupNorm 替代 BatchNorm：无 running stats，保存/加载逐位一致。

### 3.2 训练

与 CNN/ViT 相同的 `split_indices`、种子、轮数、学习率、batch；
`Trainer(model, MSELoss(), Adam(...), task='regression')`；
测试集只在最后 `predict_array` 一次。

### 3.3 评估

- 开环：MAE/RMSE/P95/R²；
- 鲁棒性：11 种扰动；
- 闭环：`sample_tracks(6,4,2)`，成功判据 |dev|≤10px；
- 噪声闭环：σ=0.05；
- 持久化：save/load 后预测 max_diff=0（GroupNorm 无 buffer）；
- 延迟：batch=1 单帧；
- 边界输入：全黑/全白/纯噪声/道路偏出。

### 3.4 三方对比

读取：

- `outputs/donkeycar/donkeycar_测试结果.json`（CNN）；
- `outputs/donkeycar_vit/donkeycar_vit_结果.json`（ViT）；
- 本脚本的结果（ResNet）。

生成 `三方对比.png`（MAE / 闭环最大偏差 / 成功率）与
`鲁棒性对比.png`（11 种扰动 × 3 模型）。

### 3.5 为什么用 ResNet

- 残差连接让 6 个 3×3 卷积块更容易优化；
- 卷积的局部性/平移等变先验适合“看路边线”的任务；
- 相比 ViT，不需要大量数据也能训练；
- 相比纯 CNN，更深的特征层次可能提升精度与鲁棒性。

## 4. 调用关系

复用 `donkeycar.generate_track_dataset` 与 `donkeycar_eval` 的
`split_indices/regression_metrics/perturb_images/closed_loop_run/sample_tracks`；
模型来自 `miniflows.resnet.build_resnet`。

## 5. 易错点

1. ResNet 训练比 CNN 慢约 2 倍（6 轮约 64 秒），完整 5 种子约 10 分钟；
2. GroupNorm 组数必须整除通道数；
3. 保存/加载只验证参数；GroupNorm 无 running stats，所以是一致的；
4. 0.2pp/0.001 级差异不能宣称显著；
5. 极端赛道 0% 是控制边界，不是模型 bug。

## 6. 自检问题

1. ResNet 与 CNN/ViT 的参数量分别多少？为什么可以对标？
2. 残差连接解决了什么问题？
3. GroupNorm 与 BatchNorm 的持久化差异？
4. 三方对比时，哪些指标必须同协议才公平？
