# `tests/test_core.py` 阅读指南

> 前置：graph/ops/layers/autodiff ｜ 难度：★★ ｜ 建议 45 分钟 ｜ 可执行规格

## 1. 文件定位

框架核心的“可执行规格”：6 组测试，覆盖节点行为、算子形状、数值梯度、
MLP/CNN 训练、summary/Dropout。全部通过输出 `✓ 核心框架全部测试通过`。

## 2. 结构速览

| 顺序 | 测试/行号 | 检查什么 |
|---|---|---|
| 1 | `test_node_basic` (51) | 运算符、梯度累加、detach、常量、拓扑 |
| 2 | `naive_conv2d` (120) / `naive_maxpool` (138) | 朴素参考实现 |
| 3 | `test_op_shapes` (149) | 形状 + 与朴素实现数值对拍 |
| 4 | `test_gradient_checks` (233) | 34 项中心差分 |
| 5 | `test_mlp_training` (385) | MBGD/Adam 损失下降 |
| 6 | `test_cnn` (401) | 小 CNN 形状 + 全参数梯度 + L2 |
| 7 | `test_summary_and_dropout` (446) | summary/Dropout/named_params/优化器 |

## 3. 逐段讲解

### 3.1 节点基础

- `np.ndarray - Node` 必须返回 Node（`__array_priority__`）；
- `x*x+x` 梯度 `2x+1`（累加）；
- 常量节点 `grad is None`；
- `build_topo` 4 节点、`collect_edges` 4 条边；
- `seq_d.train_mode` 递归。

### 3.2 形状与朴素对拍

- conv：多组 stride/padding/kernel；
- pool：k=2/3，s=1/2；
- 与 `naive_conv2d`/`naive_maxpool` 逐元素比较；
- 广播：add/mul/broadcast_add 的合法形状。

### 3.3 34 项梯度检验

包括 add/sub/mul/div/matmul/broadcast_add/exp/log/pow/sqrt/relu/sigmoid/tanh/
softmax/conv2d/maxpool/avgpool/reshape/transpose/sum/mean/MSE/BCE/CE 等。
最大相对误差约 **2.978e-9**，阈值 1e-5。

注意：这个数字与 exp2 的 17 项（1.361e-7）不是同一口径。

### 3.4 MLP 训练

2-8-1 合成回归：MBGD/Adam 初始 loss 1.932 → 0.011 左右，
证明优化器与反向链路可用。

### 3.5 小 CNN

`Conv(1→2,3,pad1)+ReLU+Pool+Flatten+Linear`，
全参数梯度 max_rel≈8.4e-10；附带 `loss+l2_penalty` 组合检验。

### 3.6 summary/Dropout

- summary 参数量与手算一致（712）；
- Dropout 训练态置零比例≈0.5、保留位缩放 2、评估态恒等；
- `named_params` 命名 `conv1.W/layer2.W`；
- MBGD step 后参数变化、lr 可调。

## 4. 运行

```bash
python tests/test_core.py
```

全部断言通过才打印成功；任何失败会抛 AssertionError 并显示具体项。

## 5. 易错点

1. 测试顺序依赖：每个测试自己设置随机种子；
2. 梯度检验输入要远离不可导点；
3. `test_cnn` 的 `Linear` 输入维度必须与 Flatten 一致；
4. 命名规则按容器索引，不是“第几个带参层”。

## 6. 自检问题

1. 34 项与 exp2 的 17 项有什么区别？
2. 如何构造一个测试，证明 `unbroadcast` 正确？
3. 为什么 Dropout 评估态必须恒等？
4. summary 的参数量如何手算校验？
