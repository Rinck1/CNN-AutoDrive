# `common_setup.py` 阅读指南

> 前置：无 ｜ 难度：★ ｜ 建议 15 分钟 ｜ 建议读 1 遍

## 1. 文件定位

所有可运行脚本的**统一环境入口**。它不参与任何算法，但决定了四件事：

1. Windows 控制台中文不乱码；
2. 随机种子固定，实验可复现；
3. matplotlib 中文字体正常、负号正常；
4. 强制 Agg 后端，无界面环境也能 `savefig`，禁止 `plt.show()`。

调用方式（每个脚本开头）：

```python
from common_setup import init_env
init_env(seed=42)
```

## 2. 结构速览

| 顺序 | 符号/行号 | 作用 |
|---|---|---|
| 1 | `init_env` (24) | 一次性完成 UTF-8、种子、字体、Agg、输出目录 |

## 3. 逐段讲解

### 3.1 UTF-8 控制台

对 `sys.stdout` / `sys.stderr` 调 `reconfigure(encoding="utf-8", errors="ignore")`。
Windows 默认 GBK，中文 print 会 `UnicodeEncodeError` 或乱码；这一步兜底。

### 3.2 随机种子

`random.seed(seed)` + `np.random.seed(seed)`。
注意：框架里 `Trainer(seed=...)` 还会控制 DataLoader 的乱序，二者不冲突。

### 3.3 matplotlib 配置

```python
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", ...]
plt.rcParams["axes.unicode_minus"] = False
```

- 字体按优先级降级；Linux 需自装中文字体；
- `axes.unicode_minus=False` 让负号用 ASCII 渲染，避免缺字方块。

### 3.4 输出目录

若传 `output_dir`，自动 `os.makedirs(..., exist_ok=True)` 并返回绝对路径。
大多数脚本还会再调用一次 `init_env(seed=42, output_dir=...)`，重复设置无害。

## 4. 调用关系

所有 `applications/`、`experiments/`、`tests/` 脚本在 import 阶段调用；
`miniflows/visualization.py` 内部还有 `_ensure_chinese_font()` 二次兜底。

## 5. 易错点

- 不要在 `common_setup` 里 import torch/sklearn：核心框架只依赖 numpy/matplotlib；
- 不要在脚本里再 `plt.show()`：Agg 后端下无窗口，且会阻塞/报错；
- 若中文字体仍显示方块，检查系统是否装了微软雅黑/黑体，或改 `font.sans-serif`。

## 6. 自检问题

1. 为什么种子要同时设置 `random` 与 `numpy`？
2. 为什么必须用 Agg 后端？
3. `axes.unicode_minus=False` 解决什么问题？
