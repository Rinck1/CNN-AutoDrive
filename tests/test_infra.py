# -*- coding: utf-8 -*-
"""
tests/test_infra.py —— 训练基础设施模块单元测试
==================================================

覆盖范围（不依赖核心模块 graph/ops/layers 的部分可直接运行）：

1. data 缩放：standard_scale / minmax_scale 统计量正确 + apply_scale 可逆/复用；
2. one_hot：形状、命中位置正确；
3. train_test_split：形状正确、分层划分的类别比例与原数据一致；
4. DataLoader：批数 = ceil(N/B)、批形状、固定种子可复现；
5. make_blobs / random_shift_images：形状与零位移恒等性；
6. EarlyStopping / LRScheduler(step+plateau)：边界行为正确；
7. save_model / load_model：往返参数完全一致；count_params 正确；
8. draw_computation_graph：对"假 Node"（鸭子类型）成功出图。

运行方式（项目根目录）：python tests/test_infra.py
全部通过打印"全部测试通过"并以 0 退出；任一失败以 1 退出。
"""

import os
import sys
import tempfile

# ------------------ 标准脚本初始化头（契约第 1 节） ------------------
try:  # Windows 控制台强制 UTF-8，避免中文乱码报错
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # 使脚本能 import miniflows / common_setup
from common_setup import init_env
init_env(seed=42)  # 统一设置：随机种子 + matplotlib 中文字体 + Agg 后端
# ---------------------------------------------------------------------

import numpy as np
import matplotlib
matplotlib.use("Agg")                                        # 无界面后端（双保险）

from miniflows.data import (standard_scale, minmax_scale, apply_scale, one_hot,
                            train_test_split, make_blobs, DataLoader,
                            random_shift_images)
from miniflows.visualization import draw_computation_graph
from miniflows.utils import (accuracy, mse, EarlyStopping, LRScheduler,
                             save_model, load_model, count_params)

TMP = tempfile.mkdtemp(prefix="miniflows_test_")             # 测试临时输出目录
_results = []                                                # 收集 (用例名, 是否通过, 信息)


def _case(name):
    """装饰器：把测试函数包装为带中文结果统计的用例。"""
    def deco(fn):
        def wrapper():
            try:
                fn()                                         # 执行测试体
                _results.append((name, True, "通过"))
                print(f"[通过] {name}")
            except Exception as e:                           # 捕获失败，继续跑后续用例
                _results.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"[失败] {name} -> {type(e).__name__}: {e}")
        return wrapper
    return deco


# ======================================================================
# 1. 数据缩放（可逆性 / 统计量复用）
# ======================================================================
@_case("standard_scale/minmax_scale 统计量与可逆性")
def test_scaling():
    rs = np.random.RandomState(0)
    X = rs.randn(40, 5) * 7 + 10                             # 构造非零均值大方差数据
    Xs, sc = standard_scale(X)                               # 标准化
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-12)       # 每列均值 ~ 0
    assert np.allclose(Xs.std(axis=0), 1, atol=1e-12)        # 每列标准差 ~ 1
    # 可逆性：(Xs * std + mean) 应还原 X（apply_scale 正向复用统计量也一致）
    assert np.allclose(Xs * sc["std"] + sc["mean"], X)
    assert np.allclose(apply_scale(X, sc), Xs)               # 同一统计量对新数据变换一致

    Xm, scm = minmax_scale(X, feature_range=(-1, 1))         # 区间缩放到 [-1,1]
    assert Xm.min() >= -1 - 1e-12 and Xm.max() <= 1 + 1e-12  # 落在目标区间
    assert np.allclose(apply_scale(X, scm), Xm)              # 复用统计量一致
    # 逆变换：从区间缩放还原
    lo, hi = scm["feature_range"]
    back = (Xm - lo) / (hi - lo) * (scm["max"] - scm["min"]) + scm["min"]
    assert np.allclose(back, X)                              # 完全可逆


# ======================================================================
# 2. one-hot 编码
# ======================================================================
@_case("one_hot 编码形状与命中位置")
def test_one_hot():
    y = np.array([2, 0, 1, 1])                               # 4 个样本 3 类
    oh = one_hot(y, 3)
    assert oh.shape == (4, 3)                                # (N,C)
    assert oh.dtype == np.float64                            # float 矩阵
    assert np.allclose(oh.sum(axis=1), 1)                    # 每行恰有一个 1
    assert oh.argmax(axis=1).tolist() == y.tolist()          # 命中位置=原标签
    assert oh[0, 2] == 1.0 and oh[0, 0] == 0.0               # 抽查首行
    try:                                                     # 越界标签应报错
        one_hot(np.array([5]), 3)
        raise AssertionError("越界标签未报错")
    except ValueError:
        pass                                                 # 预期行为


# ======================================================================
# 3. train_test_split（形状 + 分层正确性）
# ======================================================================
@_case("train_test_split 形状与分层比例")
def test_split():
    rs = np.random.RandomState(1)
    X = rs.randn(120, 4)
    # 构造不平衡数据：60/40/20（比例 0.5/0.333/0.167），检验分层是否保持比例
    y = np.array([0] * 60 + [1] * 40 + [2] * 20)
    rs.shuffle(y)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_size=0.3,
                                              shuffle=True, stratify=y, seed=0)
    n_total, n_test = len(y), len(y_te)
    assert len(X_tr) == len(y_tr) and len(X_te) == len(y_te)  # X/y 数量对齐
    assert n_total == len(y_tr) + n_test                     # 无样本丢失
    # 各类别占比（训练/测试）都应与原数据接近（分层目标）
    orig = np.bincount(y, minlength=3) / n_total
    tr_prop = np.bincount(y_tr, minlength=3) / len(y_tr)
    te_prop = np.bincount(y_te, minlength=3) / n_test
    assert np.abs(tr_prop - orig).max() < 0.02, f"训练集比例偏差过大: {tr_prop}"
    assert np.abs(te_prop - orig).max() < 0.03, f"测试集比例偏差过大: {te_prop}"
    # 不分层模式：只检查形状与可复现性
    a1 = train_test_split(X, y, test_size=5, seed=7)         # 整数 test_size
    a2 = train_test_split(X, y, test_size=5, seed=7)
    assert len(a1[3]) == 5                                   # 整数按样本数切
    assert np.array_equal(a1[0], a2[0])                      # 同种子结果一致


# ======================================================================
# 4. DataLoader（批数 / 批形状 / 可复现）
# ======================================================================
@_case("DataLoader 批数与乱序可复现")
def test_dataloader():
    X = np.arange(100 * 3, dtype=np.float64).reshape(100, 3)
    y = np.arange(100)
    dl = DataLoader(X, y, batch_size=32, shuffle=True, seed=0)
    assert len(dl) == 4                                      # ceil(100/32)=4
    batches = list(dl)                                       # 迭代一遍
    assert [len(b[0]) for b in batches] == [32, 32, 32, 4]   # 3 满批 + 1 尾批
    all_idx = np.concatenate([b[1] for b in batches])        # 标签拼回应是全量集合
    assert sorted(all_idx.tolist()) == list(range(100))      # 无丢失无重复
    # 乱序确实打乱了顺序
    assert not np.array_equal(batches[0][1], np.arange(32))
    # 固定 seed：两个全新 loader 的第一次迭代序列一致（可复现）
    dlA = DataLoader(X, y, batch_size=32, shuffle=True, seed=0)
    dlB = DataLoader(X, y, batch_size=32, shuffle=True, seed=0)
    seqA = [b[1].copy() for b in dlA]
    seqB = [b[1].copy() for b in dlB]
    assert all(np.array_equal(a, b) for a, b in zip(seqA, seqB))


# ======================================================================
# 5. make_blobs / random_shift_images
# ======================================================================
@_case("make_blobs 与 random_shift_images")
def test_blobs_shift():
    X, y = make_blobs(n_samples=150, centers=3, n_features=2, seed=3)
    assert X.shape == (150, 2) and y.shape == (150,)         # 形状正确
    assert set(np.unique(y)) == {0, 1, 2}                    # 3 类标签齐全

    imgs = np.random.rand(5, 28, 28)                         # (N,H,W) 形式
    sh = random_shift_images(imgs, max_shift=3, seed=0)
    assert sh.shape == imgs.shape                            # 形状不变
    assert random_shift_images(imgs, max_shift=0).sum() == imgs.sum()  # 零位移恒等
    imgs4 = np.random.rand(5, 1, 28, 28)                     # (N,1,H,W) 形式
    assert random_shift_images(imgs4, 2, seed=0).shape == imgs4.shape  # 形状保持
    sh1 = random_shift_images(imgs, 3, seed=1)               # 固定种子可复现
    sh2 = random_shift_images(imgs, 3, seed=1)
    assert np.array_equal(sh1, sh2)


# ======================================================================
# 6. EarlyStopping / LRScheduler 行为
# ======================================================================
class _DummyOpt:
    """最小化优化器替身：只有 LRScheduler 需要的 .lr 属性。"""
    def __init__(self, lr=0.1):
        self.lr = lr


@_case("EarlyStopping 连续无改善触发")
def test_early_stopping():
    es = EarlyStopping(patience=3, min_delta=0.01)
    seq = [es(v) for v in [1.0, 0.9, 0.85, 0.845, 0.845, 0.845]]
    # 前三轮持续改善；后三轮无改善，第 3 次无改善时触发 True
    assert seq == [False, False, False, False, False, True], seq
    assert es.best_loss == 0.85 and es.counter == 3          # 属性正确
    es2 = EarlyStopping(patience=2, min_delta=0.0)           # 恢复改善后计数清零
    s2 = [es2(v) for v in [1.0, 1.0, 0.5, 0.5, 0.5]]
    # 第 2 个 1.0 不构成改善（min_delta=0）：计数 1；0.5 刷新最优计数清零；
    # 之后两个 0.5 计数 1、2，第二次无改善（patience=2）触发 True
    assert s2 == [False, False, False, False, True], s2


@_case("LRScheduler step 与 plateau 两种模式")
def test_lr_scheduler():
    # ---------- step 模式：每 10 轮 ×0.5 ----------
    opt = _DummyOpt(0.1)
    sch = LRScheduler(opt, mode="step", step_size=10, gamma=0.5, min_lr=1e-6)
    lrs = [sch.step(epoch=ep) for ep in range(35)]           # 显式传 epoch（0 起）
    assert abs(lrs[0] - 0.1) < 1e-12 and abs(lrs[9] - 0.1) < 1e-12   # 第 1~10 轮 0.1
    assert abs(lrs[10] - 0.05) < 1e-12                       # 第 11 轮起减半
    assert abs(lrs[24] - 0.025) < 1e-12                      # 第 21 轮起再减半
    assert abs(lrs[30] - 0.0125) < 1e-12                     # 第 31 轮起第三次衰减
    assert abs(opt.lr - 0.0125) < 1e-12                      # 优化器 lr 已被更新
    # ---------- plateau 模式：无改善计数达 patience 才衰减 ----------
    opt2 = _DummyOpt(0.1)
    sch2 = LRScheduler(opt2, mode="plateau", patience=2, gamma=0.5, min_lr=1e-6)
    seq = [sch2.step(val_loss=v) for v in [1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8]]
    assert abs(seq[0] - 0.1) < 1e-12                         # 改善期不衰减
    assert abs(seq[3] - 0.05) < 1e-12                        # 连续 2 轮无改善 -> 减半
    assert abs(seq[6] - 0.025) < 1e-12                       # 再次停滞 -> 再减半
    assert abs(opt2.lr - 0.025) < 1e-12
    # ---------- min_lr 下限保护 ----------
    opt3 = _DummyOpt(1e-7)
    sch3 = LRScheduler(opt3, mode="step", step_size=1, gamma=0.5, min_lr=1e-6)
    assert abs(sch3.step(epoch=1) - 1e-6) < 1e-15            # 衰减不低于 min_lr


# ======================================================================
# 7. save/load 往返一致性 + count_params
# ======================================================================
class _Param:
    """参数替身：只带 .data（模拟 Node 的数据面）。"""
    def __init__(self, arr):
        self.data = arr


class _FakeModel:
    """模型替身：实现契约要求的 named_params() 接口。"""
    def __init__(self, init=1.0):
        self.W1 = _Param(np.full((4, 8), init))
        self.b1 = _Param(np.full((8,), init))
        self.W2 = _Param(np.full((8, 3), init))

    def named_params(self):
        return [("fc1.W", self.W1), ("fc1.b", self.b1), ("fc2.W", self.W2)]


@_case("save_model/load_model 往返一致 + count_params")
def test_save_load():
    m = _FakeModel(init=1.0)
    m.W1.data[:] = np.random.randn(4, 8)                     # 写入随机权重
    m.W2.data[:] = np.random.randn(8, 3)
    m.b1.data[:] = np.random.randn(8)
    assert count_params(m) == 4 * 8 + 8 + 8 * 3              # 参数量统计正确
    path = os.path.join(TMP, "fake_model.npz")
    save_model(m, path)                                      # 保存
    assert os.path.exists(path)                              # 文件已生成
    m2 = _FakeModel(init=0.0)                                # 全新"零初始化"模型
    load_model(m2, path)                                     # 加载
    for (n1, p1), (n2, p2) in zip(m.named_params(), m2.named_params()):
        assert n1 == n2                                      # 参数名对齐
        assert np.array_equal(p1.data, p2.data)              # 逐元素完全一致（往返一致）
    # 原地加载：m2 的参数对象引用不变（copyto 语义）
    before_id = id(m2.W1.data)
    load_model(m2, path)
    assert id(m2.W1.data) == before_id                       # .data 数组原地更新


# ======================================================================
# 8. draw_computation_graph 对假 Node 出图
# ======================================================================
class _FakeNode:
    """鸭子类型 Node：只有契约里的可视化必需属性。"""
    _counter = 0

    def __init__(self, data, op_name="Op", name="", parents=None):
        self.data = np.asarray(data)
        self.op_name = op_name
        self.name = name
        self.parents = parents or []                         # 父节点引用构成 DAG
        _FakeNode._counter += 1

    @property
    def shape(self):
        return self.data.shape


@_case("draw_computation_graph 对假 Node 出图成功")
def test_draw_graph():
    # 构造小型 MLP 形状的计算图: x,W1 -> MatMul -> +b1 -> ReLU -> MatMul W2 -> Loss
    x = _FakeNode(np.zeros((2, 4)), "Constant", "x")         # 输入
    w1 = _FakeNode(np.zeros((4, 8)), "Parameter", "W1")
    b1 = _FakeNode(np.zeros((8,)), "Parameter", "b1")
    mm1 = _FakeNode(np.zeros((2, 8)), "MatMul", "", [x, w1])
    ad1 = _FakeNode(np.zeros((2, 8)), "BroadcastAdd", "", [mm1, b1])
    rl = _FakeNode(np.zeros((2, 8)), "ReLU", "", [ad1])
    w2 = _FakeNode(np.zeros((8, 3)), "Parameter", "W2")
    mm2 = _FakeNode(np.zeros((2, 3)), "MatMul", "", [rl, w2])
    loss = _FakeNode(np.zeros(()), "CrossEntropyLoss", "loss", [mm2])
    path = os.path.join(TMP, "fake_graph.png")
    n_nodes, n_edges = draw_computation_graph(loss, path, title="假 Node 测试图")
    assert os.path.exists(path) and os.path.getsize(path) > 0  # 文件生成且非空
    assert n_nodes == 9                                      # 节点数正确
    assert n_edges == 8                                      # 边数正确（8 条 parent->child）
    # max_nodes 截断路径也应正常工作
    path2 = os.path.join(TMP, "fake_graph_small.png")
    n2, e2 = draw_computation_graph(loss, path2, title="截断测试", max_nodes=5)
    assert os.path.exists(path2) and n2 <= 5                 # 截断后仍成功保存


# ======================================================================
# 主入口：逐个运行并汇总
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MiniFlows 基础设施模块单元测试（tests/test_infra.py）")
    print("=" * 60)
    for fn in (test_scaling, test_one_hot, test_split, test_dataloader,
               test_blobs_shift, test_early_stopping, test_lr_scheduler,
               test_save_load, test_draw_graph):
        fn()                                                 # 逐用例运行（失败不中断）
    n_pass = sum(1 for _, ok, _ in _results if ok)           # 通过数
    n_fail = len(_results) - n_pass                          # 失败数
    print("-" * 60)
    for name, ok, msg in _results:                           # 汇总表
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  ({msg})"))
    print("-" * 60)
    print(f"结果: {n_pass}/{len(_results)} 通过" + ("" if n_fail == 0
                                                    else f"，{n_fail} 个失败"))
    sys.exit(1 if n_fail else 0)                             # 退出码反映测试状态
