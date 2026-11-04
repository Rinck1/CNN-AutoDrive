# `tests/test_resnet.py` 阅读指南

> 前置：`resnet.py`、`autodiff.py` ｜ 难度：★★ ｜ 建议 25 分钟

## 1. 文件定位

ResNet / GroupNorm 组件测试：

1. GroupNorm 分组统计；
2. GroupNorm 数值梯度；
3. 极小 ResNet 前向 + 全参数梯度；
4. 默认 ResNet 参数量（44,337）与保存/加载往返。

## 2. 结构速览

| 顺序 | 测试/行号 | 检查什么 |
|---|---|---|
| 1 | `test_groupnorm_stats` | 每组均值≈0、方差≈1 |
| 2 | `test_groupnorm_grad` | GroupNorm + 加权和 的梯度 |
| 3 | `test_tiny_resnet` | 16×16 极小 ResNet 全参数梯度 |
| 4 | `test_default_resnet` | 44,337 参数、64×64 前向、npz 往返 |

## 3. 逐段讲解

### 3.1 GroupNorm 统计

```python
gn = GroupNorm2D(4, num_groups=2, eps=0.0)
y = gn(Node(x)).data.reshape(2,2,2,8,8)   # (N,G,C/G,H,W)
mean ≈ 0, var ≈ 1
```

- 用 `eps=0` 验证纯归一化；默认 `eps=1e-5` 会让方差略小于 1；
- 分组轴是 `(C/G,H,W)`，不是整个 C。

### 3.2 梯度

```python
gradient_check(lambda ns: ops.sum(ops.mul(gn(ns[0]), W)), [X])
```

- 加权和让每个元素都有非平凡梯度；
- 实测 max_rel ≈ 4.2e-8。

### 3.3 极小 ResNet

```python
build_resnet(c0=4, widths=(4,8), blocks=(1,1), num_groups=4)
```

- 输入 `(2,1,16,16)` → 输出 `(2,1)`；
- 参数 1,401；
- 全参数梯度 max_rel ≈ 5.5e-8。

### 3.4 默认 ResNet

- 参数必须等于 **44,337**（与 CNN/ViT 对齐）；
- 保存/加载后预测最大差必须为 0；
- 因为 GroupNorm 没有 running stats，参数保存即可完整复现。

## 4. 运行

```bash
python tests/test_resnet.py
```

末尾输出 `✓ ResNet 全部测试通过`。

## 5. 易错点

1. GroupNorm 的 `num_groups` 必须整除通道数；
2. 统计检验用 `eps=0`，否则方差误差约 1e-6；
3. 极小 ResNet 的 `img_size` 要能被 stem stride 与 stage stride 整除；
4. 保存/加载要求参数命名一致。

## 6. 自检问题

1. 为什么 GroupNorm 不需要 running stats？
2. 44,337 参数的硬编码意义是什么？
3. 极小 ResNet 的 1,401 参数如何拆解？
