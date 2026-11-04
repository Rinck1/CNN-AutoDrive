# `miniflows/autodiff.py` 阅读指南

> 前置：`graph.py`、泰勒展开 ｜ 难度：★★★ ｜ 建议 60 分钟 ｜ 核心

## 1. 文件定位

回答一个核心问题：**你的反向传播算对了吗？**
用中心差分数值梯度作为“标准答案”，与解析梯度逐元素比较，输出相对误差报告。

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `numerical_gradient` (37) | 对每个输入数组做中心差分 |
| 2 | `_rel_error` (74) / `_max_rel` (88) | 相对误差与最大值 |
| 3 | `gradient_check` (98) | 函数级：数值 vs 解析，返回报告 |
| 4 | `_walk_layers` (149) | 递归遍历模型层 |
| 5 | `check_model_gradients` (158) | 模型级：逐参数检验 |

## 3. 逐段讲解

### 3.1 中心差分

```python
f(x+eps) - f(x-eps)
--------------------
      2*eps
```

- 截断误差 O(eps²)，比前向差分 O(eps) 精度高；
- `eps=1e-5`、float64 时，总误差约 1e-10；
- `f` 接收 `list[np.ndarray]`，返回 Python float。

### 3.2 相对误差

```python
rel = |a - n| / max(1e-8, |a| + |n|)
```

- 用相对误差而不是绝对误差：梯度量级可能从 1e-6 到 1e3；
- 分母加 1e-8 防止 0/0；
- 判据：所有元素 `rel < tol`（默认 1e-5）。

### 3.3 gradient_check 流程

1. 把输入数组包装成 `Node`（requires_grad=True）；
2. 调 `f(nodes)` 得标量 Node，`backward()` 得解析梯度；
3. 对每个输入数组调 `numerical_gradient`；
4. 逐元素算相对误差，汇总 `passed/max_rel_error/per_input`。

### 3.4 check_model_gradients 流程

1. `pred = model(Node(x_np))`，`loss = loss_fn(pred, y)`；
2. `loss.backward()` 得全部参数梯度；
3. 对 `named_params()` 中每个参数，用数值梯度扰动其 `data` 重新前向；
4. 输出每个参数的最大相对误差。

注意：数值扰动会临时改参数并恢复，必须保证模型无随机性（Dropout 用评估模式）。

## 4. 调用关系

`tests/test_core.py`、`tests/test_transformer.py`、`experiments/exp2_grad_check.py`
都调用它；应用脚本不直接调用。

## 5. 易错点

1. **不可导点**：ReLU 在 0、maxpool 并列最大值会导致误报，输入要避开；
2. **float32**：舍入误差会让数值梯度噪声达到 1e-3，必须 float64；
3. **Dropout/随机层**：检验前要 `train_mode(False)`；
4. **相对误差阈值**：1e-5 是经验值；tanh 等函数可能到 1e-7；
5. **数值扰动会改变参数**：必须用副本/恢复，不能污染训练。

## 6. 自检问题

1. 中心差分为什么比前向差分准？误差阶分别是什么？
2. 为什么用相对误差？举一个绝对误差会误判的例子；
3. 如何故意构造一个错误反向，验证 `gradient_check` 能抓住？
4. 为什么梯度检验必须 float64？
