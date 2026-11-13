# `miniflows/profiler.py` 阅读指南

> 前置：各模型的层结构 ｜ 难度：★★ ｜ 建议 40 分钟

## 1. 文件定位

课程要求“与其他框架比较训练时间、资源占用、FLOPS”。本模块提供**统一口径**的度量：

```
count_params(model)
count_flops(model, input_shape)        # FLOPs = 2×MACs
measure_latency(model, input_shape)    # 单帧/单批延迟
measure_activation_bytes(model, ...)   # 前向峰值激活内存（下界）
profile_model(model, ...)              # 汇总成 dict
```

## 2. 结构速览

| 顺序 | 符号 | 作用 |
|---|---|---|
| 1 | `_out_hw` | 卷积/池化输出尺寸 |
| 2 | `_flops_of` | 单层 FLOPs + 输出形状（递归处理复合模块） |
| 3 | `count_params` | 参数量 |
| 4 | `count_flops` | 整网 FLOPs/MACs |
| 5 | `measure_latency` | warmup + 重复计时 |
| 6 | `_iter_layers` | 递归遍历所有 Layer |
| 7 | `measure_activation_bytes` | 包装 forward 累加每层输出字节 |
| 8 | `profile_model` | 一次性返回全部指标 |

## 3. 逐段讲解

### 3.1 FLOPs 口径（最重要）

统一用 `2 × MACs`：

```
Conv   : 2·N·OH·OW·F·Cg·KH·KW     Cg=每组输入通道（分组卷积按组算）
Linear : 2·N·in·out
GroupNorm/LayerNorm : ≈5·N·元素数
Pool/激活 : ≈1 FLOP/元素
```

为什么必须统一：不同论文/框架是否计 bias、乘加算 1 还是 2 都不同。
**跨框架比较要用同一公式重算**，不能各抄各的数字。

手算校验（`tests/test_models.py`）：
`Conv(1→4,3,p1) + ReLU + Conv(4→8,3,s2,p1) + Flatten + Linear(128→10)` 在 8×8 上：
4608 + 256 + 9216 + 2560 = **16,640**，与 `count_flops` 完全一致。

### 3.2 复合模块的递归

`_flops_of` 按类型分派：

- 叶子层（Conv2D/Linear/Pool/GN/激活）直接算；
- `BasicBlock`、`MultiHeadSelfAttention`、`TransformerBlock` 单独处理；
- 有 `layers` / `blocks` / `stages` / `stem` 属性的容器递归；
- 深度可分离卷积按 `dw + pw` 两段相加。

### 3.3 激活内存

`measure_activation_bytes` 的做法：

1. 用 `_iter_layers` 递归找到所有 Layer 实例；
2. 给每个实例的 `forward` 套一层记录器（实例属性遮蔽类方法）；
3. 前向一遍，累加每层输出的 `nbytes`；
4. 结束时恢复（无实例属性则删除），避免污染模型。

这是**下界**：反向还要保存中间变量，训练内存通常是 2~3 倍。

### 3.4 延迟测量

`warmup=3`（让首次分配/缓存稳定）+ `repeat=20` 取均值与 P95。
batch=1 时才是“单帧延迟”，batch>1 时是吞吐测试。

## 4. 调用关系

`applications/mnist_web.py` 打印模型规模；`compare_resources.py` 对比两框架；
`tests/test_models.py` 校验口径。

## 5. 易错点

1. `input_shape` 不含 batch（如 `(1,64,64)`），batch 由 `batch` 参数控制；
2. FLOPs 是结构属性，同结构必然相同，不能用来“证明”实现更高效；
3. 激活内存不含梯度、不含优化器状态；
4. `measure_latency` 会切换 `train_mode(False)`，测量后不会自动恢复；
5. 自定义层如果没有 `layers/blocks` 属性，FLOPs 会返回 0（需要扩展 `_flops_of`）。

## 6. 自检问题

1. 手算一个小 conv 的 FLOPs 并与 `count_flops` 对照；
2. 为什么 FLOPs 低不代表延迟低？
3. 激活内存为什么是训练内存的下界？
4. 新增一种层时，需要在哪些地方扩展？
