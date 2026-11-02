# -*- coding: utf-8 -*-
"""
exp2_grad_check.py —— 扩展②：自动求导机制 + 数值梯度检验实验
====================================================================
实验目的：
    MiniFlows 的自动微分（反向传播）是整个框架正确性的基石。本实验对
    框架中全部核心算子以及一个完整的小型 MLP 模型做“数值梯度检验”：
    用中心差分法逐元素计算数值梯度，与反向传播得到的解析梯度比较
    相对误差，从而验证自动求导实现的正确性。

检验对象（共 16 个算子 + 1 个完整模型）：
    add / matmul / broadcast_add / mul / relu / sigmoid / tanh /
    softmax / MSELoss / BCEWithLogitsLoss / CrossEntropyLoss /
    conv2d(padding=1) / maxpool2d / avgpool2d / transpose /
    sum(axis=1, keepdims=True) 以及 MLP(4→8→3)+CrossEntropyLoss。

实现要点：
    1. 所有输入使用 float64 随机小张量（Node 内部亦统一 float64），
       保证中心差分误差可压到 1e-10 量级；
    2. relu 的输入特意构造在远离 0 的区间（|x|>=0.2），避免落在
       不可导点附近造成“数值梯度 vs 解析梯度”的伪失败；
    3. maxpool2d 的输入按“池化窗口内元素两两至少相差 0.5”构造，
       避免最大值平局/近平局（平局时数值梯度会把扰动路由到错误元素）；
    4. 对形状类算子（transpose / sum）与组合算子使用“加权和”而非
       纯求和作为标量输出，使梯度非平凡、能真正检验轴向路由是否正确。

运行方式（在项目根目录）：
    python experiments/exp2_grad_check.py

输出产物：
    outputs/exp2/梯度检验报告.md （检验表 + 结论 + 原理说明）
"""

# ====================== 标准脚本初始化头（契约第 1 节，逐字复制） ======================
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
# =====================================================================================

import numpy as np

from miniflows import ops  # 算子函数式接口（add/matmul/conv2d/...）
from miniflows.graph import constant  # 把不参与求导的常数数组包装成 Node
from miniflows.autodiff import gradient_check, check_model_gradients  # 扩展②核心工具
from miniflows.layers import Sequential, Linear, ReLU  # 组装小 MLP 用
from miniflows.losses import (MSELoss, BCEWithLogitsLoss, CrossEntropyLoss)

OUT_DIR = os.path.join(_ROOT, 'outputs', 'exp2')  # 输出目录：outputs/exp2/
os.makedirs(OUT_DIR, exist_ok=True)               # 自动创建输出目录

TOL = 1e-5   # 检验阈值：最大相对误差 < 1e-5 视为通过（契约默认值）
EPS = 1e-5   # 中心差分扰动步长（契约默认值）


# ----------------------------------------------------------------------
# 输入构造辅助函数
# ----------------------------------------------------------------------
def make_away_from_zero(shape, low=0.2, high=1.2):
    """生成“远离 0”的随机输入（专供 relu 这类存在不可导点的算子使用）。

    参数：
        shape (tuple): 输出张量形状；
        low / high (float): 幅值区间 [low, high)。

    返回：
        np.ndarray: 各元素幅值在 [0.2,1.2)、符号随机的数组。
        由于 |x| >= 0.2 远大于扰动步长 eps=1e-5，可保证差分扰动
        不会跨越 relu 在 x=0 处的不可导折点，避免伪失败。
    """
    mag = np.random.uniform(low, high, size=shape)    # 幅值 ∈ [0.2, 1.2)，远离不可导点 0
    sign = np.random.choice([-1.0, 1.0], size=shape)  # 随机正负号，覆盖两个线性区域
    return mag * sign                                  # 幅值 × 符号 = 远离 0 的随机输入


def make_pool_input(n=2, c=2, h=6, w=6, k=2):
    """构造 maxpool 数值检验专用的输入：每个池化窗口内元素两两至少相差 0.5。

    原因：maxpool 的梯度只回传到窗口内最大值位置。若窗口内两个元素
    数值极接近（差距 < 2*eps），中心差分可能把扰动“路由”到另一个元素，
    造成解析梯度与数值梯度不一致的伪失败。逐窗口无放回抽样可彻底避免。

    参数：
        n, c, h, w (int): 输入形状 (n, c, h, w)；h、w 需能被 k 整除；
        k (int): 池化窗口边长。

    返回：
        np.ndarray: 形状 (n, c, h, w)，每个 k×k 窗口内 4 个值
                    从 {0.1,0.2,...,9.9} 中无放回抽取，两两至少差 0.1。
    """
    out = np.zeros((n, c, h, w))                      # 预分配输出
    pool = np.arange(1, 100) / 10.0                   # 候选值池 {0.1,...,9.9}
    for i in range(n):                                # 逐样本
        for j in range(c):                            # 逐通道
            for r in range(h // k):                   # 逐窗口行
                for q in range(w // k):               # 逐窗口列
                    vals = np.random.choice(pool, size=k * k, replace=False)  # 无放回 → 窗口内必不平局
                    out[i, j, r * k:(r + 1) * k, q * k:(q + 1) * k] = vals.reshape(k, k)  # 填入窗口
    return out


# ----------------------------------------------------------------------
# 逐算子检验定义
# ----------------------------------------------------------------------
def build_checks():
    """构造全部算子级检验用例。

    返回：
        list[dict]: 每项含
            'name'  : 算子名（打印/表格用）
            'desc'  : 输入形状描述
            'f'     : 接收 list[Node] 返回标量 Node 的前向函数
            'arrays': 参与求导检验的输入数组列表
    说明：对形状类/组合算子统一采用“输出 × 随机权重矩阵再求和”的
    标量化方式，使每个输入元素的梯度都不平凡，能真正检验轴向路由。
    """
    checks = []

    # 1. add：逐元素加。加权和标量化，检验两个输入各自的梯度路由
    A = np.random.randn(3, 4); B = np.random.randn(3, 4); Wa = np.random.randn(3, 4)
    checks.append(dict(
        name='add', desc='a:(3,4), b:(3,4)',
        arrays=[A, B],
        f=lambda nd: ops.sum(ops.mul(ops.add(nd[0], nd[1]), constant(Wa))),  # sum((a+b)*W)
    ))

    # 2. matmul：矩阵乘。加权和对输出标量化，同时检验 dA 与 dB
    Am = np.random.randn(3, 4); Bm = np.random.randn(4, 5); Wm = np.random.randn(3, 5)
    checks.append(dict(
        name='matmul', desc='A:(3,4), B:(4,5)',
        arrays=[Am, Bm],
        f=lambda nd: ops.sum(ops.mul(ops.matmul(nd[0], nd[1]), constant(Wm))),  # sum((A@B)*W)
    ))

    # 3. broadcast_add：偏置专用广播加。重点检验“沿广播轴求和还原”（unbroadcast）
    # a 形状 (N=2, C=3, L=4)，按"通道偏置"语义 b 的长度必须等于 C=3
    # （索引 1 是通道维），框架会把它补成 (1,3,1) 再广播。
    Ab = np.random.randn(2, 3, 4); bb = np.random.randn(3); Wb = np.random.randn(2, 3, 4)
    checks.append(dict(
        name='broadcast_add', desc='a:(2,3,4), b:(3,)',
        arrays=[Ab, bb],
        f=lambda nd: ops.sum(ops.mul(ops.broadcast_add(nd[0], nd[1]), constant(Wb))),  # b 的梯度需沿广播轴求和
    ))

    # 4. mul：逐元素乘
    Au = np.random.randn(3, 4); Bu = np.random.randn(3, 4); Wu = np.random.randn(3, 4)
    checks.append(dict(
        name='mul', desc='a:(3,4), b:(3,4)',
        arrays=[Au, Bu],
        f=lambda nd: ops.sum(ops.mul(ops.mul(nd[0], nd[1]), constant(Wu))),  # sum((a*b)*W)
    ))

    # 5. relu：输入远离 0，避免不可导点伪失败
    xr = make_away_from_zero((4, 5)); Wr = np.random.randn(4, 5)
    checks.append(dict(
        name='relu', desc='x:(4,5) 远离0',
        arrays=[xr],
        f=lambda nd: ops.sum(ops.mul(ops.relu(nd[0]), constant(Wr))),
    ))

    # 6. sigmoid：光滑函数，输入拉宽到 ±3 以覆盖饱和区
    xs = np.random.randn(4, 6) * 1.5; Ws = np.random.randn(4, 6)
    checks.append(dict(
        name='sigmoid', desc='x:(4,6)',
        arrays=[xs],
        f=lambda nd: ops.sum(ops.mul(ops.sigmoid(nd[0]), constant(Ws))),
    ))

    # 7. tanh：光滑函数
    xt = np.random.randn(4, 6) * 1.5; Wt = np.random.randn(4, 6)
    checks.append(dict(
        name='tanh', desc='x:(4,6)',
        arrays=[xt],
        f=lambda nd: ops.sum(ops.mul(ops.tanh(nd[0]), constant(Wt))),
    ))

    # 8. softmax：减最大值稳定化。对“概率 × 权重”求和，检验雅可比-向量积
    xo = np.random.randn(3, 6); Wo = np.random.randn(3, 6)
    checks.append(dict(
        name='softmax', desc='x:(3,6) axis=-1',
        arrays=[xo],
        f=lambda nd: ops.sum(ops.mul(ops.softmax(nd[0], axis=-1), constant(Wo))),  # 注意：不能只 sum(softmax)，那是常数
    ))

    # 9. MSELoss：target 闭包传入（契约约定 target 为 np.ndarray，不参与求导）
    pred_ms = np.random.randn(6, 2); tgt_ms = np.random.randn(6, 2)
    mse_fn = MSELoss()
    checks.append(dict(
        name='MSELoss', desc='pred:(6,2)',
        arrays=[pred_ms],
        f=lambda nd: mse_fn(nd[0], tgt_ms),  # 标量损失 = mean((pred-target)^2)
    ))

    # 10. BCEWithLogitsLoss：数值稳定实现应处处光滑，0/1 标签两类都要有
    logits_b = np.random.randn(8, 1) * 2.0
    tgt_b = (np.random.rand(8, 1) > 0.5).astype(np.float64)
    if tgt_b.min() == tgt_b.max():          # 极小概率下全 0/全 1，则手工混入两类
        tgt_b[0, 0] = 1.0 - tgt_b[0, 0]
    bce_fn = BCEWithLogitsLoss()
    checks.append(dict(
        name='BCEWithLogitsLoss', desc='z:(8,1), t∈{0,1}',
        arrays=[logits_b],
        f=lambda nd: bce_fn(nd[0], tgt_b),
    ))

    # 11. CrossEntropyLoss：log-softmax 稳定化 + NLL
    logits_c = np.random.randn(7, 4)
    labels_c = np.random.randint(0, 4, size=7)  # 整型标签 (N,)
    ce_fn = CrossEntropyLoss()
    checks.append(dict(
        name='CrossEntropyLoss', desc='logits:(7,4)',
        arrays=[logits_c],
        f=lambda nd: ce_fn(nd[0], labels_c),
    ))

    # 12. conv2d：任务指定形状 x:(2,3,8,8), W:(4,3,3,3), padding=1（im2col 正/反向 + col2im）
    xc = np.random.randn(2, 3, 8, 8); Wc = np.random.randn(4, 3, 3, 3) * 0.5
    checks.append(dict(
        name='conv2d', desc='x:(2,3,8,8) W:(4,3,3,3) pad=1',
        arrays=[xc, Wc],
        f=lambda nd: ops.sum(ops.conv2d(nd[0], nd[1], padding=1)),  # 对输出全和标量化，检验 dx 与 dW
    ))

    # 13. maxpool2d：专用输入避免窗口内最大值平局
    xmp = make_pool_input(n=2, c=2, h=6, w=6, k=2)
    checks.append(dict(
        name='maxpool2d', desc='x:(2,2,6,6) k=2 s=2',
        arrays=[xmp],
        f=lambda nd: ops.sum(ops.maxpool2d(nd[0], kernel_size=2, stride=2)),  # 梯度只应回传到每窗口最大值
    ))

    # 14. avgpool2d：光滑（窗口内均匀分摊）
    xap = np.random.randn(2, 2, 6, 6)
    checks.append(dict(
        name='avgpool2d', desc='x:(2,2,6,6) k=2 s=2',
        arrays=[xap],
        f=lambda nd: ops.sum(ops.avgpool2d(nd[0], kernel_size=2, stride=2)),
    ))

    # 15. transpose：加权和理解轴置换后的梯度路由（axes=(2,0,1): (2,3,4)->(4,2,3)）
    xtr = np.random.randn(2, 3, 4); Wtr = np.random.randn(4, 2, 3)
    checks.append(dict(
        name='transpose', desc='x:(2,3,4) axes=(2,0,1)',
        arrays=[xtr],
        f=lambda nd: ops.sum(ops.mul(ops.transpose(nd[0], axes=(2, 0, 1)), constant(Wtr))),
    ))

    # 16. sum(axis=1, keepdims=True)：加权和对归约结果标量化，检验归约轴向
    xsm = np.random.randn(2, 5, 4); Wsm = np.random.randn(2, 1, 4)
    checks.append(dict(
        name='sum(axis=1,keepdims)', desc='x:(2,5,4) -> (2,1,4)',
        arrays=[xsm],
        f=lambda nd: ops.sum(ops.mul(ops.sum(nd[0], axis=1, keepdims=True), constant(Wsm))),
    ))

    return checks


def build_mlp_check():
    """构造完整小模型 MLP(4→8→3)+CrossEntropyLoss 的检验用例。

    返回：
        dict: 与算子用例同构（name/desc），另附执行检验所需对象。
    """
    mlp = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))  # 参数量 (4*8+8)+(8*3+3)=67
    X_mlp = np.random.randn(20, 4)                        # 小批量输入
    y_mlp = np.random.randint(0, 3, size=20)              # 三分类整型标签
    ce_fn = CrossEntropyLoss()

    # 兼容 check_model_gradients 对 loss_fn 的两种可能调用约定：
    #   loss_fn(pred) 或 loss_fn(pred, y) —— 标签一律取闭包中的 y_mlp
    def loss_fn(pred, _y=None):
        return ce_fn(pred, y_mlp)

    return dict(name='MLP(4→8→3)+CrossEntropyLoss',
                desc='X:(20,4), 参数67个', model=mlp, x=X_mlp, loss_fn=loss_fn)


# ----------------------------------------------------------------------
# 主流程：执行检验、打印中文检验表、生成 Markdown 报告
# ----------------------------------------------------------------------
def main():
    """执行全部梯度检验并输出报告。

    流程：
        1) 逐算子调用 miniflows.autodiff.gradient_check（float64、eps=1e-5、tol=1e-5）；
        2) 对 MLP(4→8→3) 调用 check_model_gradients 检验全部参数；
        3) 打印中文检验表（算子/最大相对误差/是否通过），失败项打 ✗ 但继续跑完；
        4) 生成 outputs/exp2/梯度检验报告.md。
    """
    results = []  # [(name, desc, max_rel, passed, per_input明细), ...]

    # ---------- 1. 算子级检验 ----------
    for ck in build_checks():
        res = gradient_check(ck['f'], ck['arrays'], eps=EPS, tol=TOL)  # 解析梯度 vs 中心差分数值梯度
        max_rel = float(res['max_rel_error'])                          # 全部输入、全部元素的最大相对误差
        passed = bool(res['passed'])
        results.append((ck['name'], ck['desc'], max_rel, passed, res.get('per_input', [])))
        flag = '✓' if passed else '✗'                                  # 失败也只标记，不中断，继续跑完
        print(f"  {flag} {ck['name']:<24s} 最大相对误差 = {max_rel:.3e}")

    # ---------- 2. 完整模型（MLP）检验 ----------
    mk = build_mlp_check()
    mres = check_model_gradients(mk['model'], mk['x'], mk['loss_fn'], eps=EPS)
    m_max = float(mres['max_rel_error'])
    m_pass = bool(mres['passed'])
    # 把 per_param 字典转成与算子级一致的明细结构，便于统一制表
    per_param = mres.get('per_param', {})
    per_items = [(k, float(v.get('max_rel_error', float('nan')))) for k, v in per_param.items()]
    results.append((mk['name'], mk['desc'], m_max, m_pass,
                    [{'max_rel_error': v, 'shape': k} for k, v in per_items]))
    flag = '✓' if m_pass else '✗'
    print(f"  {flag} {mk['name']:<24s} 最大相对误差 = {m_max:.3e}（共 {len(per_items)} 个参数张量）")

    # ---------- 3. 打印中文汇总表 ----------
    n_fail = sum(1 for r in results if not r[3])
    print('\n' + '=' * 78)
    print(f"梯度检验汇总表（数值梯度：中心差分 eps={EPS:g}，判定阈值 tol={TOL:g}）")
    print('=' * 78)
    print(f"{'序号':<4s}{'算子/模型':<32s}{'最大相对误差':<16s}{'结果':<6s}")
    print('-' * 78)
    for i, (name, desc, mr, ps, _) in enumerate(results, 1):
        print(f"{i:<6d}{name:<30s}{mr:<18.3e}{'✓ 通过' if ps else '✗ 未通过'}")
    print('-' * 78)
    if n_fail == 0:
        print("✓ 所有算子梯度检验通过")
    else:
        print(f"✗ 共有 {n_fail} 项未通过检验（详见上表），自动求导实现存在偏差，需要排查。")

    # ---------- 4. 生成 Markdown 报告 ----------
    write_report(results, n_fail)
    print(f"\n报告已保存至：{os.path.join(OUT_DIR, '梯度检验报告.md')}")


def write_report(results, n_fail):
    """生成 outputs/exp2/梯度检验报告.md。

    参数：
        results (list): main() 中收集的检验结果元组列表；
        n_fail (int): 未通过项数。
    """
    lines = []
    lines.append('# 扩展② 自动求导梯度检验报告\n')
    lines.append('> 生成脚本：`experiments/exp2_grad_check.py` ｜ '
                 f'数值梯度：中心差分（eps={EPS:g}）｜ 判定阈值：最大相对误差 < {TOL:g}\n')

    lines.append('## 一、检验方法：中心差分原理\n')
    lines.append(
        '对以数组 `x` 为自变量的标量函数 `f(x)`（这里即“前向计算到标量损失”），第 i 个分量的数值梯度按'
        '**中心差分**计算：\n\n'
        '```\n'
        'g_i ≈ [ f(x + ε·e_i) − f(x − ε·e_i) ] / (2ε),   ε = 1e-5\n'
        '```\n\n'
        '与反向传播得到的**解析梯度** `a_i` 逐元素比较相对误差：\n\n'
        '```\n'
        'rel_i = |a_i − n_i| / max(1e-8, |a_i| + |n_i|)\n'
        '```\n\n'
        f'当**所有输入、所有元素**的最大 rel < {TOL:g} 时判定该算子检验通过。\n')

    lines.append('## 二、检验结果\n')
    lines.append('| 序号 | 算子/模型 | 输入形状 | 最大相对误差 | 是否通过 |')
    lines.append('|---:|---|---|---:|:---:|')
    for i, (name, desc, mr, ps, _) in enumerate(results, 1):
        lines.append(f"| {i} | {name} | {desc} | {mr:.3e} | {'✓ 通过' if ps else '✗ 未通过'} |")
    lines.append('')
    if n_fail == 0:
        lines.append('**结论：✓ 所有算子梯度检验通过。** 反向传播的解析梯度与中心差分数值梯度在 '
                     f'1e-{int(np.log10(1 / TOL)) if TOL > 0 else 5}=1e-5 精度下完全一致，'
                     '自动微分链路（含广播还原、卷积 col2im、池化路由、softmax 稳定化）实现正确。\n')
    else:
        lines.append(f'**结论：✗ 共 {n_fail} 项未通过检验。** 未通过算子的解析梯度与数值梯度存在偏差，'
                     '应优先检查该算子反向闭包中的链式法则实现与广播轴还原。\n')

    lines.append('## 三、误差来源分析\n')
    lines.append(
        '1. **截断误差（O(ε²)）**：中心差分公式由泰勒展开截断而来，真实导数与差分值之差约为 '
        '(ε²/6)·f\'\'\'(ξ)。ε=1e-5 时该项约为 1e-11 量级，是本检验的主要误差来源之一。\n'
        '2. **舍入误差（O(u/ε)）**：浮点减法相消使差分分子仅保留约 u/ε 的相对精度'
        '（u 为机器精度）。ε 越小舍入误差越大，因此 ε 不能盲目取小；'
        '总误差在 ε ≈ u^(1/3) 附近最小。\n'
        '3. **不可导点**：relu 在 x=0、maxpool 在窗口内两值相等处不存在连续导数。'
        '本实验通过“输入远离 0（|x|≥0.2）”“逐窗口无放回采样（元素两两至少差 0.1）”'
        '规避不可导点/近平局，防止伪失败。\n'
        '4. **数值稳定化路径**：softmax / CrossEntropy / BCEWithLogits 均采用减最大值等'
        '稳定化写法。这类写法与原始函数数学等价且处处光滑，不影响梯度检验，'
        '但避免了 exp 溢出带来的 inf/NaN。\n'
        '5. **求和顺序差异**：解析梯度与数值梯度的浮点累加顺序不同，会引入 ~1e-16 量级的'
        '随机舍入差，远小于阈值。\n')

    lines.append('## 四、为什么必须使用 float64\n')
    lines.append(
        '- float64 的机器精度 u ≈ 2.2×10⁻¹⁶，中心差分总误差最小的步长约为 ε ≈ u^(1/3) ≈ 6×10⁻⁶，'
        '恰好与本实验使用的 eps=1e-5 同量级，此时可把相对误差压到 **1e-10 ~ 1e-12**，'
        '足以暴露反向传播中的任何实现错误；\n'
        '- 若用 float32（u ≈ 1.2×10⁻⁷），最优步长退化为 ~5×10⁻³，数值梯度噪声高达 1e-3 量级，'
        '比判定阈值 1e-5 还大，检验将失去分辨力；\n'
        '- 因此契约规定 `Node` 内部数据统一转 `np.float64`（数据集可用 float32，进 Node 时自动转换），'
        '这正是为梯度检验服务的设定。\n')

    text = '\n'.join(lines)
    with open(os.path.join(OUT_DIR, '梯度检验报告.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)


if __name__ == '__main__':
    main()
