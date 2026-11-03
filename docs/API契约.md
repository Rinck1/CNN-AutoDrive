# MiniFlows 框架 API 契约（v1.0）

> 本文档是全体并行开发的**唯一接口规范**。所有模块、脚本必须严格按此签名实现与调用。
> 框架名：**MiniFlows**（迷你流），纯 numpy 实现自动微分的迷你深度学习框架。

## 0. 总则

- **依赖**：核心框架仅依赖 `numpy`；可视化依赖 `matplotlib`；数据下载可用 `torchvision`（有降级方案）。
- **数据类型**：`Node` 内部数据统一转 `np.float64`（保证数值梯度检验精度）；数据集本身可用 float32，进 Node 时自动转换。
- **计算图模型**：无全局 Graph 对象。计算图 = `Node` 节点 + `parents` 引用构成的 DAG。每个非叶子节点持有 `_backward` 闭包；从损失节点调用 `backward()` 时按拓扑排序逆序执行。
- **中文注释（最高优先级要求）**：每个模块文件头有中文说明块；每个类/函数必须有中文 docstring（功能/参数/返回值/实现思路）；关键代码行有行内中文注释。
- **风格**：简单易用，仿 Keras（`Sequential([...])` → `Trainer.fit` 风格）。

## 1. 标准脚本初始化头（每个可运行脚本开头直接复制）

```python
# -*- coding: utf-8 -*-
import os
import sys
try:  # Windows 控制台强制 UTF-8，避免中文乱码报错
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # 使脚本能 import miniflows / common_setup
from common_setup import init_env
init_env(seed=42)  # 统一设置：随机种子 + matplotlib 中文字体 + Agg 后端
```

- 所有脚本**在项目根目录运行**：`python applications/xxx.py`、`python experiments/xxx.py`。
- 输出保存到 `outputs/<脚本主题>/` 子目录（用 `os.makedirs(..., exist_ok=True)` 自动创建）。
- **禁止 `plt.show()`**（Agg 后端，只用 `savefig`）。
- `common_setup.init_env(seed=42, output_dir=None)`：已实现于项目根 `common_setup.py`（勿改），设置种子/中文字体/Agg 后端/可选输出目录。

## 2. miniflows/graph.py —— 节点与计算图

```python
class Node:
    """计算图节点。"""
    def __init__(self, data, requires_grad=False, name=None):
        # data: np.ndarray 或标量 -> 统一 np.asarray(data, dtype=np.float64)
        # requires_grad: True 表示可训练参数
        # name: 参数名/调试名（可视化用）
    data: np.ndarray          # 前向值
    grad: np.ndarray | None   # 反向得到的梯度
    requires_grad: bool
    parents: list[Node]       # 父节点（输入），构成计算图 DAG
    op_name: str              # 产生本节点的算子名，如 'MatMul'；叶子参数为 'Parameter'，常量为 'Constant'
    name: str
    @property
    def shape -> tuple
    def backward(self, grad=None)   # 从本节点开始反向传播；grad=None 时种子为全 1；拓扑排序逆序执行各节点 _backward
    def zero_grad(self)             # grad 置 None（或全 0）
    def detach(self) -> Node        # 返回脱离计算图的新 Node
    # 运算符重载：__add__/__radd__/__sub__/__rsub__/__mul__/__rmul__/__truediv__/__rtruediv__/__neg__/__pow__/__matmul__
    #   与 python 数字 / np.ndarray 运算时，自动包装为常量 Node
    # 方法：sum(axis=None, keepdims=False)、mean(axis=None, keepdims=False)、max()、
    #       exp()、log()、sqrt()、reshape(shape)、flatten(start_axis=1)、transpose(axes=None)

def constant(array, name=None) -> Node   # requires_grad=False
def param(array, name=None) -> Node      # requires_grad=True, op_name='Parameter'
def build_topo(root) -> list[Node]       # 拓扑序（输入在前，root 最后）
def collect_edges(root) -> list[tuple]   # [(parent, child, op_name), ...] 供可视化取边
```

反向实现要点：梯度累加用 `+=`（节点被多次使用时不丢失）；`requires_grad=False` 的节点可不参与反向但不影响正确性。

## 3. miniflows/ops.py —— 算子（函数式接口，闭包注册 `_backward`）

所有函数输入 `Node`（或自动包装的数组/标量），返回新 `Node`；`op_name` 建议取 Pascal 命名（如 `'MatMul'`）。

```python
# —— 基础运算 ——
add(a, b)                       # 逐元素加，支持广播；反向需对广播轴求和还原
sub(a, b); mul(a, b); div(a, b) # 逐元素
matmul(a, b)                    # 矩阵乘 @
broadcast_add(a, b)             # b 形状为 (F,) 或 (1,F) 或 (1,F,1,1)，自动广播加（偏置专用）
neg(x); exp(x); log(x); pow(x, n)   # n 为标量

# —— 激活函数（课程要求三种 + 额外 softmax）——
relu(x)                         # max(0,x)，反向 (x>0)*g
sigmoid(x)                      # 数值稳定写法；可复用 exp
tanh(x)                         # np.tanh
softmax(x, axis=-1)             # 减最大值稳定化

# —— 卷积 / 池化 ——
conv2d(x, W, b=None, stride=1, padding=0)
    # x:(N,C,H,W)  W:(F,C,KH,KW)  b:(F,) 或 (1,F)  -> (N,F,OH,OW)
    # 实现：im2col 展开为两个大矩阵乘（性能关键！），反向 dW/dx 用 col2im
maxpool2d(x, kernel_size=2, stride=2)   # 反向只把梯度放到前向最大值位置（需记录位置索引）
avgpool2d(x, kernel_size=2, stride=2)

# —— 形状 / 归约 ——
reshape(x, shape); flatten(x, start_axis=1); transpose(x, axes=None)
sum(x, axis=None, keepdims=False); mean(x, axis=None, keepdims=False)

# —— 工具 ——
unbroadcast(grad, shape)        # 把 grad 沿广播轴求和还原到 shape
im2col(x, kh, kw, stride, padding) / col2im(col, x_shape, kh, kw, stride, padding)  # 供 conv 使用与测试
```

## 4. miniflows/layers.py —— 层与容器

```python
class Layer:
    def forward(self, x: Node) -> Node
    def __call__(self, x) -> Node          # 调 forward
    def params(self) -> list[Node]         # 可训练参数（默认 []）
    def named_params(self) -> list[tuple]  # [(层内参数名如 'W', Node), ...]
    def train_mode(self, flag: bool)       # 训练/评估模式切换（Dropout 用；默认 pass）
    def zero_grad(self)

class Linear(in_features, out_features, init='he'|'xavier'|'normal', init_std=0.05, name=None)
    # W:(in,out) He 初始化 std=sqrt(2/in)（默认），xavier std=sqrt(2/(in+out))，normal 用 init_std
    # b:(out,)=0；前向 = x @ W + b

class Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, init='he', name=None)
    # W:(F,C,KH,KW) He 初始化 std=sqrt(2/(C*KH*KW))；前向调用 ops.conv2d

class MaxPool2D(kernel_size=2, stride=2)
class AvgPool2D(kernel_size=2, stride=2)
class Flatten()                 # (N,C,H,W) -> (N, C*H*W)
class ReLU(); class Sigmoid(); class Tanh(); class Softmax(axis=-1)
class Dropout(p=0.5)            # train_mode(True)：mask=(rand>p)/(1-p) 缩放；评估：恒等

class Sequential(*layers):
    # forward/__call__ 逐层传递
    # params() / named_params()  -> 名字形如 'layer0.W'（有 name 属性用 name）
    # train_mode(flag) 递归设置全部层
    # zero_grad() 对全部参数
    # summary(input_shape)  打印每层名/输出形状/参数量/总参数量（中文表格）
```

## 5. miniflows/losses.py —— 损失函数

```python
class MSELoss:
    def __call__(self, pred: Node, target: np.ndarray) -> Node   # 标量：mean((pred-target)^2)

class BCEWithLogitsLoss:
    def __call__(self, logits: Node, target01: np.ndarray) -> Node
    # 数值稳定实现：max(z,0) - z*t + log(1+exp(-|z|)) 的均值；z:(N,1) 或 (N,)，t∈{0,1}

class CrossEntropyLoss:
    def __call__(self, logits: Node, labels_int: np.ndarray) -> Node
    # logits:(N,C)，labels:(N,) 整型；内部 log-softmax（减 max 稳定）+ NLL 均值
    # 反向闭包：(softmax_onehot - onehot) / N

def l2_penalty(model, lam) -> Node
    # lam * Σ ||W||² / 2，对 model.named_params() 中所有参数（可只对 W 类参数），返回标量 Node
```

## 6. miniflows/optim.py —— 优化器

```python
class Optimizer:
    def __init__(self, params: list[Node], lr: float)
    def zero_grad(self)             # 所有 p.grad 置 None
    def step(self)                  # 用 p.grad 更新 p.data（直接改 .data，不建图）

class MBGD(Optimizer):              # 小批量梯度下降（Mini-Batch Gradient Descent）
    def __init__(self, params, lr=0.01, momentum=0.0)   # momentum>0 时启用动量（扩展⑤）

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    # m/v 与参数同形；t 步数计数；更新 m_hat/(sqrt(v_hat)+eps)，含偏差修正
```

## 7. miniflows/autodiff.py —— 自动求导与数值梯度检验工具（扩展②）

```python
def numerical_gradient(f, arrays: list[np.ndarray], eps=1e-5) -> list[np.ndarray]
    # f: 接收 list[np.ndarray] 返回标量 float；对每个数组做中心差分数值梯度

def gradient_check(f, arrays, eps=1e-5, tol=1e-5) -> dict
    # f: 接收 list[Node] 返回标量 Node；流程：
    #   1) 包装为 Node 前向 + backward() 得解析梯度
    #   2) numerical_gradient 得数值梯度
    #   3) 逐元素相对误差 rel = |a-n| / max(1e-8, |a|+|n|)
    # 返回 {'passed': bool(所有输入最大 rel<tol), 'max_rel_error': float,
    #        'per_input': [{'max_rel_error','shape'}, ...]}

def check_model_gradients(model, x_np, loss_fn, eps=1e-5) -> dict
    # 对完整模型：loss = loss_fn(model(x), y)，对全部参数做数值检验
    # 返回 {'passed', 'max_rel_error', 'per_param': {参数名: {...}}}
```

## 8. miniflows/data.py —— 数据预处理模块（扩展④）

```python
def standard_scale(X) -> (X_scaled, scaler)      # scaler={'mean':..., 'std':...}
def minmax_scale(X, feature_range=(0,1)) -> (X_scaled, scaler)
def apply_scale(X, scaler) -> X_scaled           # 用训练集统计量变换测试集
def one_hot(y_int, num_classes) -> np.ndarray    # (N,) -> (N,C) float
def train_test_split(X, y, test_size=0.2, shuffle=True, stratify=None, seed=0)
    # -> (X_train, y_train, X_test, y_test)；stratify=y 时按类别比例分层抽样
def make_blobs(n_samples=200, centers=2, center_box=(-3.0,3.0), std=1.0, n_features=2, seed=0)
    # -> (X:(n,2), y:(n,)) 合成高斯团簇（逻辑回归用，不依赖 sklearn）

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True, seed=None)
    def __iter__(self)   # yield (xb, yb) 两个 np.ndarray（不包装 Node，由 Trainer 包装）
    def __len__(self)    # 批数

def load_iris() -> (X:(150,4) float64, y:(150,) int, class_names: list[str])
    # 三级降级：sklearn.datasets.load_iris → 本地缓存 csv → 合成数据（打印警告）

def load_mnist(train=True, data_dir='data') -> (X:(N,1,28,28) float32 [0,1], y:(N,) int64)
    # torchvision 可用则用之；否则 urllib 从镜像下载 4 个 idx.gz 缓存到 data/mnist/；
    # 全部失败时抛出带中文指引的 RuntimeError

def load_mnist_arrays(subset_train=None, subset_test=None, data_dir='data')
    # -> (X_train, y_train, X_test, y_test)；subset 参数用于快速模式取样（前 N 个，类别尽量均衡：按类别轮流取样）
    # 已确认镜像 https://ossci-datasets.s3.amazonaws.com/mnist/ 可达

def random_shift_images(X, max_shift=2, seed=0) -> np.ndarray
    # 图像平移数据增强（整数像素循环平移），X:(N,1,H,W) 或 (N,H,W)
```

## 9. miniflows/visualization.py —— 可视化（扩展① + 通用画图）

```python
def setup_chinese_font()
    # rcParams: font.sans-serif=['Microsoft YaHei','SimHei',...]; axes.unicode_minus=False
    # （common_setup.init_env 已调用，脚本通常无需再调）

def draw_computation_graph(root: Node, path: str, title='计算图', max_nodes=200, figsize=None)
    # 纯 matplotlib 画 DAG（不依赖 graphviz）：
    # 布局：按拓扑深度分层（x=深度，y=同层均匀分布）
    # 节点框：显示 name / op_name + shape；有向箭头 parent -> child
    # 参数节点与损失节点可用不同颜色区分；保存 path png 并关闭 figure

def plot_loss_curve(history: dict, path, title='训练曲线')
    # history 形如 {'train_loss':[...], 'val_loss':[...], 'val_acc':[...]}；有啥画啥，双 y 轴（loss 左 acc 右）

def plot_lines(curves: dict[str, list], xlabel, ylabel, title, path)
    # 通用多曲线对比图（图例=键名）

def plot_images_grid(images, titles=None, path, n=8, cols=4)
    # images:(N,H,W)/(N,1,H,W)；灰度 cmap；每张图下方显示 titles[i]
```

## 10. miniflows/trainer.py —— 训练器

```python
class Trainer:
    def __init__(self, model, loss_fn, optimizer, task='classification', seed=0)
        # task: 'classification' 或 'regression'
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=10, batch_size=32, shuffle=True,
              callbacks=None, l2_fn=None, verbose=True) -> dict
        # history = {'train_loss': [...], 'val_loss': [...], 'val_acc': [...], 'lr': [...]}
        # 每个 epoch：DataLoader 分批 → batch 转 Node → pred=model(x)
        #   → loss = loss_fn(pred, yb) (+ l2_fn() 若提供) → loss.backward()
        #   → optimizer.step(); optimizer.zero_grad()
        # batch 前 model.train_mode(True)；验证前 model.train_mode(False)
        # callbacks: list[fn(state: dict) -> bool]，state 含 epoch/val_loss/train_loss 等，返回 True 则提前停止
    def evaluate(self, X, y, batch_size=256) -> (avg_loss, acc)
        # classification：acc=argmax 命中率（输出维度为 1 时按 >0.5 判 1，配合 BCEWithLogitsLoss）
        # regression：acc 位置返回 MSE
    def predict(self, X, batch_size=256) -> np.ndarray          # 原始输出（logits/回归值）
    def predict_classes(self, X, batch_size=256) -> np.ndarray  # argmax 类别（回归时=predict）
```

## 11. miniflows/utils.py —— 框架增强功能（扩展⑤）

```python
def set_seed(seed)                      # random + numpy 种子
def accuracy(logits_or_preds, y) -> float   # 分类准确率（>1 维取 argmax；1 维按 >0.5）
def mse(pred, y) -> float
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4)
    def __call__(self, val_loss) -> bool    # 连续 patience 轮无改善(min_delta)返回 True
    # 属性：best_loss, counter
class LRScheduler:
    def __init__(self, optimizer, mode='step'|'plateau', step_size=10, gamma=0.5,
                 patience=3, min_lr=1e-6)
    def step(self, epoch=None, val_loss=None) -> float   # 更新 optimizer.lr 并返回新 lr
    # step 模式：每 step_size 轮 lr*=gamma；plateau：val_loss patience 轮无改善 lr*=gamma
def save_model(model, path)             # np.savez：named_params() 全部参数
def load_model(model, path)             # 原地加载（按名字对应 copy 到 p.data）
def count_params(model) -> int          # 参数总量
```

## 12. 输出与产物约定

| 产物 | 路径 |
|---|---|
| 扩展① 计算图 | `outputs/exp1/` |
| 扩展② 梯度检验报告 | `outputs/exp2/`（含 梯度检验报告.md） |
| 扩展③ 训练分析 | `outputs/exp3/`（含 分析结论.md） |
| 扩展④ 预处理演示 | `outputs/exp4/` |
| 扩展⑤ 优化功能演示 | `outputs/exp5/` |
| 线性/逻辑回归/鸢尾花 | `outputs/linear_regression/`、`outputs/logistic_regression/`、`outputs/iris/` |
| MNIST（自研 vs PyTorch） | `outputs/mnist/`（含 mnist_result.json）、`outputs/pytorch_benchmark/` |
| DonkeyCar | `outputs/donkeycar/` |
| MNIST 原始数据缓存 | `data/mnist/` |

## 13. 建议默认超参（快速模式，保证几分钟内跑完）

- MNIST 自研 CNN：subset_train=8000、subset_test=2000、epochs=3、batch=64、Adam lr=1e-3 → 预期测试准确率 ≥ 90%。
- 网络：Conv(1→6,5) → ReLU → MaxPool2 → Conv(6→16,5) → ReLU → MaxPool2 → Flatten(→256) → Linear(256→120) → ReLU → Linear(120→84) → ReLU → Linear(84→10)。
- PyTorch LeNet-5 与上面结构完全一致（对比才公平），同样默认快速子集。
- 鸢尾花 MLP：4→16(ReLU)→3，Adam lr=1e-3，epochs≈100（全量 150 条很快）。
- 若 numpy 版 CNN 单轮超过 8 分钟，把默认子集降到 5000/1000 并在注释中说明。


## 14. miniflows/transformer.py —— Vision Transformer 扩展

> 在核心框架之上新增的 ViT 组件，全部由既有算子组合实现，反向自动求导。
> 新增算子（ops）：`gelu(x)`、`concat(nodes, axis)`、`broadcast_to(x, shape)`、
> `select(x, index, axis)`；`matmul` 支持 `(N,H,L,Dh)` 批量维。

```python
class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5, name=None)
    # forward: y = gamma*(x-mean)/sqrt(var+eps)+beta（沿最后一维）
    # named_params -> [('gamma',Node), ('beta',Node)]

class GELU(Layer):
    # tanh 近似：0.5x(1+tanh(k(x+0.044715x^3)))，k=sqrt(2/pi)

class MultiHeadSelfAttention(Layer):
    def __init__(self, dim, heads, name=None)
    # x:(N,L,D) -> (N,L,D)；Q/K/V/O 各一个 Linear
    # last_attn: 最近一次前向的注意力权重 (N,H,L,L)，供可视化

class TransformerBlock(Layer):
    def __init__(self, dim, heads, mlp_ratio=2, name=None)
    # Pre-LN: x = x + Attn(LN(x)); x = x + MLP(LN(x))

class PatchEmbedding(Layer):
    def __init__(self, in_channels=1, patch_size=8, dim=64, name=None)
    # (N,C,H,W) -> (N,L,D)，L=(H/p)(W/p)

class ViTRegressor(Layer):
    def __init__(self, img_size=64, patch_size=8, dim=64, depth=3,
                 heads=4, mlp_ratio=2, in_channels=1, name='vit')
    # forward: patch_embed -> [CLS]+pos -> blocks -> LN -> CLS -> Linear -> Tanh
    # attention_maps(index=0) -> (N,H,L+1,L+1)（需先 forward 一次）

def build_vit(img_size=64, patch_size=8, dim=48, depth=2,
              heads=4, mlp_ratio=2, in_channels=1) -> ViTRegressor
    # DonkeyCar 默认配置：44,353 参数（与 CNN 44,426 基本对齐）
```

**broadcast_add 对齐规则补充**：`(F,)` 偏置对 4D 卷积特征图按通道维
补成 `(1,F,1,1)`；对 3D 序列 `(N,L,D)` 及更高维按最后一维补成 `(1,...,1,F)`。
