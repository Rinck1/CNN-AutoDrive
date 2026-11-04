# `tests/test_infra.py` 阅读指南

> 前置：data/trainer/utils ｜ 难度：★ ｜ 建议 30 分钟

## 1. 文件定位

基础设施的 9 个单元测试：缩放、one-hot、划分、DataLoader、合成数据/平移、
早停、学习率调度、保存/加载、计算图绘图。

## 2. 结构速览

| 顺序 | 测试/行号 | 检查什么 |
|---|---|---|
| 1 | `_case` (53) | 测试装饰器/记录器 |
| 2 | `test_scaling` (72) | 标准化可逆、统计量 |
| 3 | `test_one_hot` (95) | 形状与命中位置 |
| 4 | `test_split` (114) | 形状、分层、无重叠 |
| 5 | `test_dataloader` (142) | 批数、全覆盖、乱序可复现 |
| 6 | `test_blobs_shift` (165) | make_blobs、平移 |
| 7 | `test_early_stopping` (191) | patience/min_delta |
| 8 | `test_lr_scheduler` (205) | step/plateau 轨迹 |
| 9 | `test_save_load` (250) | npz 往返、count_params |
| 10 | `test_draw_graph` (290) | 假 Node 出图 |

## 3. 逐段讲解

### 3.1 缩放

- `standard_scale` 后均值≈0、方差≈1；
- `apply_scale` 用同一组统计量；
- 可逆性：`x*std+mean` 还原。

### 3.2 划分

- 训练/测试形状正确；
- 分层比例接近 `test_size`；
- 训练/测试索引无交集（用集合验证）。

### 3.3 DataLoader

- 批数 = `ceil(n/batch)`；
- 一个 epoch 覆盖全部样本（不丢不重）；
- 相同 seed 的乱序一致。

### 3.4 早停/调度

- EarlyStopping 连续 patience 轮无改善才触发；
- LRScheduler step 每 step_size 轮降 gamma；
- plateau 在平台期后降，且不低于 min_lr。

### 3.5 保存/加载

用 `_FakeModel`（两个参数）验证：
保存→新建模型→加载→参数逐一相等；`count_params` 正确。

### 3.6 绘图

用 `_FakeNode`（data/parents/op_name/shape）构造小图，
确认 `draw_computation_graph` 能出图且不报错。

## 4. 运行

```bash
python tests/test_infra.py
```

末尾输出 `结果: 9/9 通过`。

## 5. 易错点

1. 分层划分在样本太少时可能失败；
2. DataLoader 的 seed 与全局 seed 不要混；
3. EarlyStopping 的 `min_delta` 是“严格更小”才算改善；
4. Fake 类只需实现被调用的接口，不要过度设计；
5. 测试不要依赖输出目录里已有的文件。

## 6. 自检问题

1. 如何验证分层划分没有数据泄漏？
2. DataLoader 一个 epoch 覆盖全部样本的判据？
3. EarlyStopping 的 patience 计数何时清零？
4. 保存/加载测试为什么要用 FakeModel 而不是真实模型？
