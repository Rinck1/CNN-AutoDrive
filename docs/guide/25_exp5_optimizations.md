# `experiments/exp5_optimizations.py` 阅读指南

> 前置：optim/utils/trainer ｜ 难度：★★ ｜ 建议 40 分钟 ｜ 扩展⑤

## 1. 文件定位

框架优化功能的五组对照实验：

a) 优化器对比；b) Dropout 正则；c) 学习率调度；d) 早停；e) 保存/加载一致性。

## 2. 结构速览

| 符号/行号 | 作用 |
|---|---|
| `sec_a_optimizers` (55) | MBGD vs +momentum vs Adam |
| `_iris_subset` (103) | 鸢尾花小子集 |
| `sec_b_dropout` (132) | 有无 Dropout 的 train/val |
| `sec_c_lr_scheduler` (183) | step/plateau 的 lr 轨迹 |
| `sec_d_early_stopping` (231) | 300 轮预算下的早停 |
| `_predict` (289) | ViT/Sequential 通用推理 |
| `sec_e_save_load` (299) | 保存→加载→预测一致 |

## 3. 逐段讲解

### 3.1 优化器对比

- 三维线性回归，真值 w=[1.5,-2,0.7]，b=0.5；
- MBGD / MBGD+momentum / Adam，80 轮；
- 三者都收敛到真值附近；动量前期更快；
- 图 `优化器对比.png`。

### 3.2 Dropout

- 60 条鸢尾花，42 训练 / 18 验证；MLP 4-64-3；
- 无 Dropout：train 0.0018 / val 0.1238；
- 有 Dropout：train 0.0138 / val 0.1463；
- 小数据上 Dropout 不一定提升，结论要如实写。

### 3.3 LRScheduler

- step：每 10 轮 lr×0.5；
- plateau：验证损失 3 轮无改善才降；
- 打印 lr 轨迹并画图；
- plateau 最终降到 9.8e-5。

### 3.4 EarlyStopping

- 300 轮预算，patience=8；
- 第 37 轮最优，第 45 轮停止；
- 节省约 255 轮计算。

### 3.5 保存/加载

- 训练模型 A，保存 npz；
- 新初始化模型 B，加载；
- 加载前预测差异 9.42，加载后 **0.0**；
- 证明按参数名对号入座正确。

## 4. 产物

`outputs/exp5/`：

- 优化器对比、Dropout、学习率调度、早停、保存加载 5 张图；
- `优化功能演示结论.md`；
- `model_demo.npz`（被 `.gitignore` 排除）。

## 5. 易错点

1. Dropout 在小数据上可能变差，不要预设“一定更好”；
2. `EarlyStopping` 不恢复最优权重，最终模型是停止时的模型；
3. `LRScheduler` 直接改 `opt.lr`，不要新建优化器；
4. 保存/加载要求参数名一致；
5. Sequential 没有 `predict`，用 `_predict` 或 Trainer。

## 6. 自检问题

1. 动量为什么能加速？与 Adam 的一阶矩区别？
2. Dropout 的 inverted scaling 是什么？
3. step 与 plateau 调度分别适合什么场景？
4. 为什么保存/加载后预测必须逐位一致？
