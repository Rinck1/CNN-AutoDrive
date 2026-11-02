# MiniFlows —— 自研迷你深度学习框架 + CNN（课程综合设计）

> **MiniFlows（迷你流）**：一个纯 numpy 实现反向传播自动微分的迷你深度学习框架，
> 用它从零训练 MLP / CNN 完成鸢尾花分类、MNIST 手写数字识别等任务，
> 并与 PyTorch 做同结构基准对比。
>
> 仓库内**全中文注释**：每个模块文件头有说明块，每个类/函数有中文 docstring，
> 关键代码行有行内中文注释，方便课程答辩逐行讲解。

---

## 1. 项目简介与评分目标

| 课程要求 | 本项目实现 | 对应位置 |
|---|---|---|
| 自研深度学习框架 | 计算图 `Node` + 算子闭包反向传播 + `Sequential` 容器 + `Trainer` | `miniflows/` |
| CNN 图像分类 | LeNet-5 风格 CNN 在 MNIST 上训练（自研 numpy 实现） | `applications/`、`pytorch_benchmark/` |
| 经典模型 | 线性回归、逻辑回归、鸢尾花 MLP | `applications/` |
| 扩展① 计算图可视化 | 纯 matplotlib 分层 DAG 绘图（不依赖 graphviz） | `miniflows/visualization.py`、`experiments/exp1_graph_viz.py` |
| 扩展② 梯度检验 | 数值梯度 vs 解析梯度逐元素相对误差报告 | `miniflows/autodiff.py`、`experiments/exp2_*` |
| 扩展③ 训练分析 | 学习曲线 / 梯度范数 / 参数分布诊断 | `experiments/exp3_*` |
| 扩展④ 数据预处理与增强 | 标准化 / one-hot / 分层划分 / MNIST 归一化 / 随机平移增强 | `miniflows/data.py`、`experiments/exp4_preprocess_demo.py` |
| 扩展⑤ 框架优化功能 | 动量、Adam、Dropout、L2、早停、学习率调度、模型保存加载 | `miniflows/utils.py`、`experiments/exp5_optimizations.py` |
| 工程质量 | 统一 API 契约 + 单元测试 + 全中文注释 + 一键复现脚本 | `docs/API契约.md`、`tests/` |

**设计理念**：仿 Keras 的极简用法——`Sequential([...])` 搭网络、
`Trainer(model, loss, optim).fit(X, y)` 训练，其余一切（建图、反向传播、
模式切换）由框架自动完成。

---

## 2. 目录结构

```
综设CNN/
├── README.md                    # 本文件（项目主页）
├── requirements.txt             # 依赖清单
├── common_setup.py              # 公共环境初始化（中文/种子/Agg，勿改）
├── docs/
│   └── API契约.md               # 全体模块的唯一接口规范 v1.0
├── miniflows/                   # ===== 自研框架本体（纯 numpy）=====
│   ├── graph.py                 # Node 节点 / 拓扑排序 / backward（核心）
│   ├── ops.py                   # 算子层：matmul/conv2d/relu/softmax...
│   ├── layers.py                # Linear/Conv2D/Dropout/Sequential 容器
│   ├── losses.py                # MSE / BCE / CrossEntropy / L2 正则
│   ├── optim.py                 # MBGD(+momentum) / Adam 优化器
│   ├── autodiff.py              # 数值梯度检验工具（扩展②）
│   ├── data.py                  # 预处理与增强：缩放/one-hot/分层划分/MNIST/iris（扩展④）
│   ├── visualization.py         # 计算图可视化 + 训练曲线（扩展①）
│   ├── trainer.py               # Keras 风格训练器（history/callbacks）
│   └── utils.py                 # 早停/学习率调度/模型保存加载（扩展⑤）
├── experiments/                 # ===== 五大扩展项演示脚本 =====
│   ├── exp1_graph_viz.py        # 扩展①：计算图可视化
│   ├── exp2_gradient_check.py   # 扩展②：梯度检验（如有）
│   ├── exp3_training_analysis.py# 扩展③：训练分析（如有）
│   ├── exp4_preprocess_demo.py  # 扩展④：预处理与增强演示
│   └── exp5_optimizations.py    # 扩展⑤：优化功能演示
├── applications/                # ===== 应用任务脚本 =====
├── pytorch_benchmark/           # PyTorch 同结构 CNN 对比
├── tests/
│   └── test_infra.py            # 基础设施单元测试（9 个用例）
├── outputs/                     # 所有实验产物（图片/报告/模型）
│   ├── exp1/  exp2/  exp3/  exp4/  exp5/
│   ├── linear_regression/  logistic_regression/  iris/
│   ├── mnist/  pytorch_benchmark/  donkeycar/
└── data/                        # 数据集缓存（自动创建）
    └── mnist/                   # MNIST idx.gz 缓存（首次运行自动下载）
```

---

## 3. 环境安装

**Python 3.10+（Windows / Linux 均可）**，推荐先建虚拟环境：

```bash
# 可选：创建虚拟环境
python -m venv .venv
# Windows 激活
.venv\Scripts\activate
# 安装依赖（核心框架只需 numpy + matplotlib，其余为数据/对比实验服务）
pip install -r requirements.txt
```

依赖说明（见 `requirements.txt`）：

| 包 | 必需性 | 用途 |
|---|---|---|
| numpy | 必需 | 框架核心（计算图 / 反向传播全部基于 numpy） |
| matplotlib | 必需 | 计算图可视化与训练曲线 |
| scikit-learn | 推荐 | 鸢尾花数据集首选来源（有降级方案） |
| torchvision | 可选 | MNIST 下载首选来源（有 urllib 镜像降级） |
| torch | 可选 | 仅 PyTorch 对比实验需要 |

> 即便只有 numpy + matplotlib，框架本身与全部演示仍可运行
> （iris / MNIST 加载均有三级降级方案，见 [常见问题](#7-常见问题-faq)）。

---

## 4. 快速开始

所有命令**均在项目根目录执行**。首次运行会自动下载数据并缓存到 `data/`。

### 4.1 单元测试（约 10 秒）

```bash
python tests/test_infra.py
```
预期输出末尾：
```
结果: 9/9 通过
```

### 4.2 扩展①：计算图可视化（约 10 秒）

```bash
python experiments/exp1_graph_viz.py
```
预期输出：打印表达式 `z=(a+b)*(b+c)` 的手算梯度验证（dz/da=4, dz/db=9, dz/dc=5）
与两个图的节点/边数量；生成 `outputs/exp1/计算图_表达式.png`、`outputs/exp1/计算图_MLP.png`。

### 4.3 扩展④：数据预处理与增强演示（约 1~3 分钟，首次含 MNIST 下载约 11MB）

```bash
python experiments/exp4_preprocess_demo.py
```
预期输出：鸢尾花标准化箱线图、one-hot 示例表、分层划分分布对比、
MNIST 像素统计、平移增强对比网格，全部保存到 `outputs/exp4/`，
并生成 `outputs/exp4/预处理演示结论.md`。

### 4.4 扩展⑤：框架优化功能演示（约 1~2 分钟）

```bash
python experiments/exp5_optimizations.py
```
预期输出：优化器对比、Dropout 正则、学习率调度、早停、保存/加载一致性
五组实验的图片（`outputs/exp5/*.png`）与中文结论报告
`outputs/exp5/优化功能演示结论.md`。

### 4.5 应用任务（鸢尾花 / MNIST 等，视脚本而定）

```bash
python applications/xxx.py     # 具体脚本名以 applications/ 目录为准
```
MNIST 快速模式（子集 8000 训练 / 3 轮 Adam）在普通 CPU 上约 5~10 分钟，
预期测试准确率 ≥ 90%。

### 4.6 PyTorch 同结构对比

```bash
python pytorch_benchmark/xxx.py
```
输出 `outputs/pytorch_benchmark/`，与自研 CNN 逐项对比准确率与耗时。

---

## 5. 五大扩展项与评分标准对照

| 扩展项 | 功能 | 演示命令 | 产物 | 评分要点自评 |
|---|---|---|---|---|
| ① 计算图可视化 | 拓扑分层布局 + 节点框（名称/算子/形状）+ 有向箭头 + 语义配色，纯 matplotlib | `python experiments/exp1_graph_viz.py` | `outputs/exp1/*.png` | 手算表达式图 + 完整 MLP 图，节点/边数量打印验证 |
| ② 梯度检验 | 中心差分数值梯度 vs 解析梯度，逐元素相对误差 | `python experiments/exp2_*.py` | `outputs/exp2/梯度检验报告.md` | 全部算子 max_rel_error < 1e-5 |
| ③ 训练分析 | 学习曲线 / 梯度范数 / 参数分布诊断 | `python experiments/exp3_*.py` | `outputs/exp3/分析结论.md` | 定位过拟合/欠拟合/学习率不当 |
| ④ 数据预处理与增强 | z-score/MinMax、one-hot、分层划分、MNIST 归一化、随机平移增强 | `python experiments/exp4_preprocess_demo.py` | `outputs/exp4/*.png` + 结论 md | 前后对比图 + 量化统计 |
| ⑤ 框架优化功能 | 动量、Adam、Dropout、早停、学习率调度、模型持久化 | `python experiments/exp5_optimizations.py` | `outputs/exp5/*.png` + `优化功能演示结论.md` | 每项有对照实验 + 保存/加载往返一致 |

---

## 6. MiniFlows API 速查表

### 6.1 搭网络 + 训练（Keras 风格）

```python
from miniflows.layers import Sequential, Linear, ReLU, Dropout
from miniflows.losses import CrossEntropyLoss, l2_penalty
from miniflows.optim import Adam
from miniflows.trainer import Trainer
from miniflows.utils import EarlyStopping, LRScheduler, save_model, load_model

model = Sequential(
    Linear(4, 16, name="fc1"), ReLU(),
    Dropout(p=0.5),                 # 训练期随机失活，评估自动关闭
    Linear(16, 3, name="fc2"),
)
loss_fn = CrossEntropyLoss()
opt = Adam(model.params(), lr=1e-3)
es = EarlyStopping(patience=8)                       # 连续 8 轮无改善即停
lr_sched = LRScheduler(opt, mode="plateau", patience=3)

trainer = Trainer(model, loss_fn, opt, task="classification", seed=42)
history = trainer.train(
    X_train, y_train, X_val, y_val,
    epochs=100, batch_size=32,
    callbacks=[
        lambda s: es(s["val_loss"]),                            # 早停回调
        lambda s: lr_sched.step(val_loss=s["val_loss"]) or False,  # 调度回调
    ],
    l2_fn=lambda: l2_penalty(model, 1e-4),       # L2 正则（零参可调用）
    verbose=True,
)
acc = trainer.evaluate(X_test, y_test)[1]            # (平均损失, 准确率)
pred = trainer.predict_classes(X_test)               # argmax 类别
```

### 6.2 计算图与自动微分

```python
from miniflows.graph import param, constant
a, b, c = param(2.0, name="a"), param(3.0, name="b"), param(1.0, name="c")
z = (a + b) * (b + c)          # 运算符重载自动建图
z.backward()                   # 拓扑排序逆序执行 _backward
print(a.grad, b.grad, c.grad)  # 4.0 9.0 5.0（与手算一致）
```

### 6.3 数据加载与预处理

```python
from miniflows.data import (load_iris, load_mnist_arrays, train_test_split,
                            standard_scale, apply_scale, one_hot,
                            DataLoader, random_shift_images)

X, y, names = load_iris()                          # sklearn -> 本地缓存 -> 合成 三级降级
X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_size=0.3,
                                          stratify=y, seed=0)  # 分层抽样
X_tr_s, scaler = standard_scale(X_tr)              # 只用训练集统计量
X_te_s = apply_scale(X_te, scaler)                 # 防数据泄露
loader = DataLoader(X_tr_s, y_tr, batch_size=32, shuffle=True, seed=0)

(Xtr, ytr, Xte, yte) = load_mnist_arrays(subset_train=8000, subset_test=2000)
Xtr_aug = random_shift_images(Xtr, max_shift=2, seed=0)        # ±2 像素平移增强
```

### 6.4 计算图可视化与画图

```python
from miniflows.visualization import (draw_computation_graph, plot_loss_curve,
                                     plot_lines, plot_images_grid)
logits = model(X); loss = loss_fn(logits, y)
draw_computation_graph(loss, "outputs/graph.png", title="MLP 计算图")  # 纯 matplotlib
plot_loss_curve(history, "outputs/curve.png")                      # loss 左轴 acc 右轴
plot_images_grid(Xte[:8], titles=[f"标签 {v}" for v in yte[:8]],
                 path="outputs/mnist_grid.png")                    # 图像网格
```

### 6.5 模型持久化与参数统计

```python
from miniflows.utils import save_model, load_model, count_params
save_model(model, "outputs/model.npz")             # np.savez: 参数名 -> 数组
load_model(model, "outputs/model.npz")             # 按名对号入座原地加载
print(count_params(model))                         # 参数总量
```

---

## 7. 常见问题 FAQ

**Q1：图中中文标题显示成方块"口口"？**
A：脚本开头务必调用 `common_setup.init_env()`（会设置微软雅黑/黑体字体）。
若直接使用 `miniflows.visualization`，其画图函数内置了字体兜底（检测到默认
西文字体时自动切换中文字体）。Linux 服务器需安装中文字体（如 `fonts-noto-cjk`）。

**Q2：Windows 控制台中文输出乱码或报 UnicodeEncodeError？**
A：每个脚本开头的"标准初始化头"已强制 `sys.stdout.reconfigure(encoding='utf-8')`；
也可先执行 `chcp 65001` 把控制台切到 UTF-8 代码页。

**Q3：MNIST 下载失败怎么办？**
A：三级方案：
1. 保持网络重试（首选 torchvision，失败后自动改用镜像
   `https://ossci-datasets.s3.amazonaws.com/mnist/` 下载 4 个 idx.gz，约 11MB）；
2. 手动下载 `train-images-idx3-ubyte.gz` 等 4 个文件放入 `data/mnist/`；
3. 仍失败会抛出带中文指引的 `RuntimeError`——鸢尾花相关实验不受影响。

**Q4：鸢尾花数据加载报 sklearn 相关错误？**
A：`load_iris()` 会自动降级：sklearn → 本地缓存 `data/iris_cache.csv` →
合成高斯团簇数据（控制台打印警告），实验流程永不中断。

**Q5：训练太慢？**
A：MNIST 使用 `load_mnist_arrays(subset_train=8000, subset_test=2000)` 快速模式；
仍慢可降到 5000/1000。所有实验脚本默认参数都保证在几分钟内跑完。

**Q6：想保存/复现实验结果？**
A：全局种子由 `init_env(seed=42)` 统一固定；`Trainer(seed=...)` 控制 DataLoader
乱序；`save_model/load_model` 支持参数落盘，往返后预测逐位一致（见扩展⑤ e 小节）。

**Q7：如何运行单元测试？**
A：`python tests/test_infra.py`，全部通过输出 `结果: 9/9 通过` 并以 0 退出。

---

## 8. 输出产物索引

| 目录 | 内容 |
|---|---|
| `outputs/exp1/` | 扩展① 计算图（表达式 / MLP） |
| `outputs/exp2/` | 扩展② 梯度检验报告 |
| `outputs/exp3/` | 扩展③ 训练分析结论 |
| `outputs/exp4/` | 扩展④ 预处理 5 张对比图 + `预处理演示结论.md` |
| `outputs/exp5/` | 扩展⑤ 优化功能 5 张图 + `优化功能演示结论.md` + `model_demo.npz` |
| `outputs/linear_regression/` `logistic_regression/` `iris/` | 经典任务结果 |
| `outputs/mnist/` `outputs/pytorch_benchmark/` | 自研 vs PyTorch 对比（含 mnist_result.json） |
| `data/mnist/` | MNIST 原始数据缓存（首次运行自动生成） |
