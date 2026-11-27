# DonkeySim 实景 RGB 端到端转向模型：数据采集、闭环问题诊断、数据增强与最终效果记录

> **项目**：MiniFlows / CNN-AutoDrive  
> **日期**：2026-11-27  
> **核心模型**：纯 NumPy 自研框架 MiniFlows · RGB 转向回归 CNN（43,933 参数，CPU 单帧推理 ~0.36 ms）  
> **目标场景**：Unity DonkeySim 三赛道（`generated_road`、`mini_monaco`、`warehouse`）

---

## 一、背景与动机：为什么现有 Web 合成赛道闭环无法直接套用

项目早期在 `applications/donkeycar.py` 与 `applications/web_studio.py` 中实现的自动驾驶训练和网页闭环仿真，使用的是**脚本实时渲染的 64×64 灰度合成几何图**（黑底、灰路、白车道线，像素归一化到 `[0, 1]`）。

当将早期灰度模型直接迁移到本机 Unity 版 **DonkeySim 真实 3D 彩色相机画面**时，由于图像域（Domain）存在巨大差异（真实光照、天空、草地/沙漠/墙体纹理、透视畸变），原灰度模型在 DonkeySim 测试集上的表现几乎失效：

| 模型 | 输入形状 | 测试集 MAE（4,249 帧） | 测试集 RMSE | 大转弯方向准确率（\|steer\|≥0.10） |
|---|---|---:|---:|---:|
| 零转向基线（一直直行） | — | 0.1819 | 0.2417 | 0.00% |
| 原灰度 CNN（`donkeycar_model.npz`） | `(1, 64, 64)` | 0.2272 | 0.2824 | 43.73% |
| 原灰度评测版（`donkeycar_eval_model.npz`） | `(1, 64, 64)` | 0.2791 | 0.3506 | 38.27% |
| **新 RGB CNN v1（`steering_model.npz`）** | **`(3, 40, 64)`** | **0.0178** | **0.0358** | **99.77%** |

同理，针对 DonkeySim 实景 RGB 相机训练的新模型也**不能直接套用到原网页的 64×64 黑白合成赛道闭环**。因此本次工作的核心目标是：
1. 在 DonkeySim 真实三赛道上采集彩色图像与专家控制标签，训练纯 MiniFlows RGB 转向模型；
2. 将新模型直接接入本机 DonkeySim（TCP `127.0.0.1:9091`）跑真实闭环，并在 Web Studio 中构建实时画面与遥测看板；
3. 针对真实闭环中暴露的跑偏/冲出赛道问题进行系统诊断、数据增强与 DAgger 纠偏迭代，直至三场景全部稳定通过闭环测试。

---

## 二、初版基础数据采集与模型训练（v1）

### 2.1 数据采集架构（`applications/collect_donkeysim_rgb.py`）

- **通信方式**：通过原生 TCP Socket 直连本机 DonkeySim（`127.0.0.1:9091`），无 `gym_donkeycar` 外部依赖，保留同一遥测包内的原始 RGB JPEG 与车辆状态（`pos_x, pos_z, yaw, speed, cte, hit, time`），避免图像与标签错帧。
- **专家标签来源（`sim_centerline_controller`）**：
  - 读取赛道中心线节点（Unity 坐标除以 `WORLD_SCALE = 8.0` 对齐遥测坐标）；
  - 使用 **Pure Pursuit（纯追踪）** 控制器计算前视目标点（前视距离 `clip(0.60 + 0.30 * speed, 0.60, 1.50)`）对应的曲率与转向角：
    $$\kappa = \frac{2 \sin(\text{bearing})}{\max(\sqrt{\Delta x^2 + \Delta z^2}, 0.15)}, \quad \text{steering} = \text{clip}\!\left(\frac{\arctan(0.28 \kappa)}{16^\circ}, -1, 1\right)$$
- **多回合采集与去重**：
  - 每个场景采集 13 个独立驾驶回合（`episode_0001` ~ `episode_0013`），交错选取赛道 6%~90% 的不同起点；
  - 起点轻微扰动：`offset ∈ [-0.065, +0.065]`，`yaw_offset ∈ [-5°, +5°]`；
  - 对每帧 JPEG 计算 SHA-256 摘要，保证 30,000 帧无完全重复图像。

### 2.2 基础数据集统计（`data/donkeysim_rgb/three_scenes_10000/`）

| 场景 | 帧数 | 回合数 | 分辨率与格式 | 固定油门 | 转向标签范围 | \|steer\|≥0.10 帧数 | 速度范围 |
|---|---:|---:|---|---:|---|---:|---|
| `generated_road` | 10,000 | 13 | 160×120 RGB JPEG | 0.10 | `[-0.423, +0.436]` | 5,938 | `0.08 ~ 1.27` |
| `warehouse` | 10,000 | 13 | 160×120 RGB JPEG | 0.10 | `[-0.406, +0.925]` | 4,919 | `0.08 ~ 1.26` |
| `mini_monaco` | 10,000 | 13 | 160×120 RGB JPEG | 0.10 | `[-0.953, +0.840]` | 5,978 | `0.08 ~ 1.26` |
| **合计** | **30,000** | **39** | **160×120 RGB JPEG** | **0.10** | **`[-0.953, +0.925]`** | **16,835** | **`0.08 ~ 1.27`** |

- **防泄露划分**：按完整驾驶回合（`episode`）而非随机帧划分，约 **70% 训练（27 回合，21,502 帧）/ 15% 验证（6 回合，4,249 帧）/ 15% 测试（6 回合，4,249 帧）**。

### 2.3 图像预处理与网络结构（`applications/donkeycar_rgb.py`）

1. **预处理管线（`prepare_frame`）**：
   - 裁掉原始 `160×120` 图像顶部 `1/6`（去除无关天空与远景，保留下方 `5/6` 路面视野）；
   - 双线性缩放至 `64×40`（`CHW = (3, 40, 64)`）；
   - 像素零中心化：`pixel / 255.0 - 0.5`（值域 `[-0.5, +0.5]`）。
2. **网络结构**：
   ```text
   Input (3, 40, 64)
     → Conv2D(3 → 8, k=3, s=2, p=1) + ReLU    # (8, 20, 32)
     → Conv2D(8 → 12, k=3, s=2, p=1) + ReLU   # (12, 10, 16)
     → Conv2D(12 → 16, k=3, s=2, p=1) + ReLU  # (16, 5, 8)
     → Flatten (640)
     → Linear(640 → 64) + ReLU
     → Linear(64 → 1, init='xavier') + Tanh   # 输出 [-1, 1] 转向角
   ```
   - **参数量**：`43,933`
   - **计算量**：`780,993` FLOPs（0.78 MFLOPs）
   - **CPU 单帧推理延迟**：`0.36 ms`（纯 NumPy MiniFlows）
3. **v1 训练配置**：Adam（`lr=0.001`，65% 轮次后衰减为 `0.0004`），`batch_size=64`，`epochs=10`，MSELoss，训练耗时 `119.16 s`。
   - **v1 离线测试结果**：MAE = `0.0178`，RMSE = `0.0358`，P95 误差 = `0.0567`，大转弯方向准确率 = `99.77%`。

---

## 三、实景闭环测试暴露的问题与根因诊断

尽管 v1 模型在离线测试集上表现极佳（MAE 仅 `0.0178`），但首次接入本机 DonkeySim 跑真实闭环时，车辆很快偏离赛道甚至冲出路面。我们编写诊断脚本，在 DonkeySim 实时运行中将每一帧的 **模型预测转向（`model_steer`）** 与 **Pure Pursuit 专家真实转向（`expert_steer`）** 逐帧比对，精确定位出 **3 个算法/数据根因** 与 **3 个工程通信问题**。

### 3.1 算法与数据层根因

#### 问题 1：训练集存在系统性右偏置（Bias Drift）
从 `generated_road` 赛道中线（`route_err = 0.001`）正常发车并运行 150 步，逐帧对比如下：

| 步数 | `model_steer` | `expert_steer` | 偏差 `diff` | `cte` | `route_err` | 现象分析 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `+0.100` | `+0.026` | **`+0.075`** | `+0.300` | `0.001` | 刚发车即明显偏右 |
| 10 | `+0.069` | `-0.002` | **`+0.072`** | `+0.005` | `0.000` | 直路持续右打 |
| 60 | `-0.129` | `-0.175` | **`+0.046`** | `+0.144` | `0.022` | 左弯能转，但转向不足 |
| 90 | `-0.005` | `-0.044` | **`+0.039`** | `+0.114` | `0.014` | 持续存在 `+0.04` 系统右偏 |
| 120 | `+0.108` | `-0.033` | **`+0.142`** | `+0.263` | `0.031` | 累积右偏接近路沿 |
| 135 | `+0.105` | `-0.078` | **`+0.184`** | `+0.546` | `0.068` | **偏右后应左打拉回，模型反而继续右打** |
| 150 | `+0.409` | `+0.063` | **`+0.346`** | `+0.953` | `0.114` | **彻底失控冲出赛道** |

- **根因**：检查 v1 的 `metrics.json` 发现，整个基础数据集的转向均值为 **`+0.0500`**，训练集转向均值为 **`+0.0532`**（右弯比例偏多），且 v1 训练未做左右水平镜像翻转增强，导致模型学到了约 `+0.04 ~ +0.07` 的系统性右偏置。

#### 问题 2：模仿学习分布偏移（Covariate Shift）与纠偏（Recovery）样本缺失
- 原采集脚本 `collect_donkeysim_rgb.py` 的起点偏移仅有 `[-0.065, +0.065]`，且只要 `route_err > 0.35` 就直接中断采集。专家全程几乎完美贴着中线行驶，导致：
  1. 离线测试集也全都是贴中线的画面，因此离线 MAE 低至 `0.0178` 极具迷惑性；
  2. 在真实闭环中，一旦微小右偏累积使车靠近路沿，画面就进入模型**从未见过的外偏分布**；此时模型不但无法大幅反打救车，反而输出同向错误转向，迅速冲出赛道（尤其是无围墙的 `generated_road` 沙漠赛道）。

#### 问题 3：默认出生点偏离赛道中心
- 在 `generated_road` 场景调用默认发车时，实测第 1 步 `cte = -4.497`、`route_err = 0.562`（已远超路宽，专家需要 `+1.000` 满舵拉回），而基础训练集里 `generated_road` 的最大转向仅为 `+0.436`，模型开局仅输出 `+0.136` 无法拉回。

### 3.2 工程与通信层问题及修复

1. **已在赛道内的握手协议兼容**：当模拟器未停在菜单而是已在某赛道中时，新 TCP 连接首条消息为 `car_loaded` 或 `telemetry`，而非 `scene_selection_ready`。若直接发送 `exit_scene` 并在同一连接等待菜单消息会导致超时。
   - **修复**：在 `donkeysim_rgb_runner.py` 中实现 `_connect_and_enter_scene()`——检测到已在赛道内且需切换场景时，先刹停并发送 `exit_scene` → 关闭旧连接 → 等待 1.5 s 重新建连 → 干净接收 `scene_selection_ready` 后再 `load_scene`，彻底杜绝旧赛道缓存帧串包。
2. **`sock.makefile()` 超时永久损坏流**：Python 标准库对 `sock.makefile('rb')` 调用 `readline()` 一旦触发超时，后续读取会永久抛出 `OSError: cannot read from timed out object`。
   - **修复**：在 `donkeysim_rgb_runner.py` 中实现基于 `sock.recv(65536)` 与内部 `bytearray` 缓冲的安全按行读取器 `_SocketLineReader`，单次超时不会破坏底层 Socket。
3. **发车中线对齐保护**：启动闭环时若检测到当前赛道默认出生点严重偏离中线（`route_err > 0.30`），自动调用 `restart_at_node()` 将车辆平稳放置于赛道中线起点。

---

## 四、针对性数据增强与 DAgger 纠偏迭代（v2 → v3）

针对上述三大算法根因，我们从 **训练增强** 与 **纠偏采样** 两个维度进行改进：

### 4.1 水平镜像翻转增强（Horizontal Flip Augmentation）
在 `applications/donkeycar_rgb.py` 中新增 `--flip-aug`：
- 对训练集中的每一张图像做水平镜像翻转：`x_flip = x[:, :, :, ::-1]`；
- 对应转向标签严格取反：`y_flip = -y`；
- **作用**：训练集规模直接翻倍，且训练集转向均值从 `+0.0532` 严格归零至 **`0.00000`**，从数学上彻底消除系统性左右偏置。

### 4.2 双模式纠偏与 DAgger 采集（`applications/collect_donkeysim_recovery.py`）
设计混合采样脚本，专门覆盖“偏离中线 → 拉回中线”的状态空间：

1. **Recovery（大偏移强制恢复）模式（50% 回合）**：
   - 在赛道随机节点以较大横向偏移 `offset ∈ [-0.32, +0.32]` 和较大车头偏角 `yaw_offset ∈ [-22°, +22°]` 发车；
   - 由专家 `Route.follow()` 全力拉回中线（并叠加极小探索噪声 `N(0, 0.03)`），放宽中断阈值至 `route_err > 0.90`，完整记录大幅救车过程。
2. **DAgger（模型自驾 + 专家实时纠偏标注）模式（50% 回合）**：
   - 从中轻度偏移 `offset ∈ [-0.18, +0.18]`、`yaw_offset ∈ [-10°, +10°]` 出发，由**当前已训练的模型自己控制车辆**（若 `route_err > 0.45` 则按 `0.5 * model + 0.5 * expert` 混合防立即出界），让车辆真实进入它容易漂移的状态；
   - **每一帧的训练标签均由专家 `Route.follow()` 实时标注**“此刻正确的纠偏转向”。
3. **高价值帧优先采样**：当 `route_err ≥ 0.12` 或 `|expert_steering| ≥ 0.25` 时每步必存，中线直行段隔步抽样，显著提高纠偏样本的信息密度。

### 4.3 两阶段迭代过程

- **v2 迭代（单场景纠偏验证）**：
  - 在 `mini_monaco` 采集 2,400 帧 DAgger + 纠偏数据（`data/donkeysim_rgb/recovery_dagger_v1/`），以 `extra_repeat=3` 合入原训练集并开启 `--flip-aug`（总训练帧数 **57,404**，8 轮）。
  - **验证发现**：`mini_monaco` 闭环最大偏移直接砍半（`0.427 → 0.188`），偏离 `+0.22` 发车也能立即稳稳拉回；但尚未补采纠偏数据的 `generated_road` 沙漠赛道依然容易在随机弯道冲出路面。
- **v3 迭代（全三场景纠偏闭环定型）**：
  - 补采 `generated_road` 2,000 帧（`recovery_generated_road/`）与 `warehouse` 2,000 帧（`recovery_warehouse/`），三场景合计新增 **6,400 帧**高质量纠偏/DAgger 样本；
  - 将原训练集（21,502 帧）+ 三场景纠偏数据（6,400 × 2 = 12,800 帧）合并并开启 `--flip-aug`，最终训练集共 **68,604 帧**（转向均值 `-0.00000`），训练 8 轮（总耗时 `288.27 s`，第 6 轮后学习率由 `0.001` 降至 `0.0004`）。

---

## 五、最终效果全面对比

### 5.1 离线测试集指标演进（同一 6 个未见测试回合，4,249 帧）

| 模型版本 | 训练样本构成 | 训练集转向均值 | 最优 Val MSE | Test MAE | Test RMSE | P95 绝对误差 | 大转弯方向准确率 |
|---|---|---:|---:|---:|---:|---:|---:|
| 原灰度 CNN | 64×64 合成赛道 | — | — | 0.2272 | 0.2824 | 0.5397 | 43.73% |
| RGB v1（基线） | 21,502（无翻转/无纠偏） | `+0.0532` | 0.00087 | 0.0178 | 0.0358 | 0.0567 | 99.77% |
| RGB v2（单场景纠偏+翻转） | 57,404（含 mini_monaco 纠偏） | `0.0000` | 0.00078 | 0.0167 | 0.0338 | 0.0497 | 99.92% |
| **RGB v3（三场景纠偏+翻转）** | **68,604（含三场景 6,400 纠偏）** | **`0.0000`** | **0.00043** | **0.0162** | **0.0289** | **0.0474** | **100.00%** |

- 相比 v1 基线，v3 的验证集 MSE **降低 50.6%**（`0.00087 → 0.00043`），测试集 RMSE 降低 **19.3%**，大转弯方向准确率达到 **100.00%**。

### 5.2 真实 DonkeySim 闭环对比

#### (1) 同起点 300 步纠偏能力对照（v1 vs v2，`mini_monaco`）

| 测试条件 | v1 最大偏移 `max_route_err` | v2 最大偏移 `max_route_err` | 改善幅度 |
|---|---:|---:|---|
| 赛道中线发车（`offset = 0.00`） | 0.427 | **0.188** | 最大偏移降低 **56.0%** |
| 路边偏离发车（`offset = +0.22`） | 0.424 | **0.219** | 发车即拉回中线，不再继续外漂 |

#### (2) 最终版 v3 三场景 400 步真实闭环验收（`outputs/donkeycar_rgb/three_scenes_steering/closed_loop_eval.json`）

固定油门 `throttle = 0.10`，每场景连续运行 400 步：

| 场景 | 完成步数 | 最大 \|CTE\| | 行驶距离 | 实景闭环帧率 | 耗时 | 碰撞状态 | 结束原因 |
|---|---:|---:|---:|---:|---:|---|---|
| `generated_road` | **400 / 400** | **0.3420** | 19.15 m | 13.5 fps | 29.54 s | `none` | `steps_limit` |
| `mini_monaco` | **400 / 400** | **1.9271** | 18.88 m | 16.3 fps | 24.53 s | `none` | `steps_limit` |
| `warehouse` | **400 / 400** | **1.4687** | 18.90 m | 17.1 fps | 23.39 s | `none` | `steps_limit` |

三个场景全部达成 **400/400 步 0 碰撞通过**，其中此前极易冲进沙漠的 `generated_road` 最大 `|CTE|` 控制在 `0.342` 以内。

---

## 六、Web Studio 实景闭环集成与复现方式

### 6.1 Web 实时闭环看板
在 `applications/web_studio.py` 与 `applications/web_studio_ui.html` 中完成全链路接入：
- **后端接口**：`GET /api/donkeysim/scenes`（探测模拟器与场景）、`GET /api/donkeysim/status`、`POST /api/donkeysim/start`、`POST /api/donkeysim/stop`，并通过 SSE `donkeysim` 事件实时推送原始 JPEG 与遥测数据；同时将 `donkeycar_rgb` 注册为模型预设支持侧边栏切换与 3D 架构查看。
- **前端界面（「实时驾驶」页顶部）**：提供场景选择、油门/步数/CTE 保护配置、原始 RGB 相机实时画面（含顶部 `1/6` 裁剪参考线）、转向角指示条、2D 实时行驶轨迹画布与转向/CTE 波形图。

### 6.2 关键文件索引
- 基础三场景采集：`applications/collect_donkeysim_rgb.py`
- DAgger / 大偏移纠偏采集：`applications/collect_donkeysim_recovery.py`
- RGB 模型训练与增强：`applications/donkeycar_rgb.py`
- 本机 DonkeySim 闭环驱动：`applications/donkeysim_rgb_runner.py`
- Web 后端与实时看板：`applications/web_studio.py`、`applications/web_studio_ui.html`
- 最终模型权重与评测：`outputs/donkeycar_rgb/three_scenes_steering/steering_model.npz`、`metrics.json`、`closed_loop_eval.json`、`training.png`

### 6.3 一键复现命令

```powershell
# 1. 启动 Web Studio（先打开本机 DonkeySim 模拟器，默认端口 9091）
python web_app.py
# 浏览器访问 http://127.0.0.1:8000/#drive

# 2. 命令行独立跑实景闭环测试
python applications/donkeysim_rgb_runner.py --scene generated_road --steps 400 --throttle 0.10
python applications/donkeysim_rgb_runner.py --scene mini_monaco --steps 400 --throttle 0.10
python applications/donkeysim_rgb_runner.py --scene warehouse --steps 400 --throttle 0.10

# 3. 复现 v3 模型训练（原三场景 + 三场景纠偏数据 + 水平镜像翻转）
python applications/donkeycar_rgb.py train `
  --extra-manifest data/donkeysim_rgb/recovery_dagger_v1/samples.csv `
  --extra-manifest data/donkeysim_rgb/recovery_generated_road/samples.csv `
  --extra-manifest data/donkeysim_rgb/recovery_warehouse/samples.csv `
  --extra-repeat 2 --flip-aug --epochs 8 --batch-size 64 --lr 0.001
```

---

## 七、当前已知局限与后续改进方向

在真实 DonkeySim 闭环试跑过程中，当前模型在常规弯道与直路上已能稳定保持在车道内，但仍存在以下明确边界，留待后续迭代解决：

1. **交叉路线 / 十字路口（岔路口）出现明显偏移**
   - **现象**：当赛道出现交叉路段、视觉上重叠的近景/远景道路或十字路口时，车辆会出现明显横向偏移甚至走偏。
   - **原因**：当前模型是**单帧无导航条件的纯视觉端到端回归**（只看当前 1 帧 `3×40×64` 图像，没有历史帧时序信息，也没有“直行/左转/右转”的高层路由指令）。在交叉路口画面中存在多个合理出口，MSE 回归损失在面对多模态歧义时倾向于输出折中值，导致车辆偏离预定路线。
   - **后续方向**：引入条件模仿学习（Conditional Imitation Learning，将高层导航指令或局部路网方向作为额外输入）及多帧堆叠 / 时序网络（TCN/GRU），消除交叉口歧义。

2. **路面障碍物无法识别与避让**
   - **现象**：遇到赛道上的静止或动态障碍物（如锥桶、杂物、其他车辆）时，模型不会主动绕行或减速刹停，而是继续按车道中心线行驶并可能发生碰撞。
   - **原因**：① 专家控制器 `Route.follow()` 仅依据静态中心线节点计算转向，不感知障碍物，因此训练集中不存在避障行为样本；② 当前模型仅预测单维 `steering`，油门为固定常量（`throttle = 0.10`），不具备自主减速/刹车能力。
   - **后续方向**：在感知侧引入障碍物检测/语义分割或深度估计分支，在控制侧改为 `(steering, throttle, brake)` 多输出联合控制，并构建含避障专家策略的闭环训练集。

