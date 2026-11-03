# -*- coding: utf-8 -*-
"""
donkeycar_vit.py —— DonkeyCar 自动驾驶的 ViT 版本（与 CNN 同协议对比）
======================================================================

模型
----
    ViT-Tiny：64×64 图像 → patch 8×8（64 个 patch）→ 48 维嵌入
    → [CLS] + 位置编码 → 2 个 Pre-LN Transformer Block（4 头）
    → LayerNorm → 取 [CLS] → Linear → Tanh → 转向角 ∈ [-1,1]

    参数量 44,353，与 CNN 版（44,426）几乎完全一致，便于公平比较。

测试协议（与 applications/donkeycar_eval.py 完全一致）
------------------------------------------------------
    - 900 条合成赛道图，70%/15%/15% 训练/验证/测试；
    - 测试集全程不参与训练/早停/模型选择；
    - 5 个随机种子（0~4），每个模型 12 轮，Adam lr=1e-3，batch=32；
    - 开环指标：MAE/RMSE/P95/R²；
    - 鲁棒性：11 种图像扰动；
    - 闭环：6 条同分布 + 4 条分布外 + 2 条极端赛道，另做噪声闭环；
    - 保存/加载、单帧推理延迟、边界输入；
    - 额外：可视化第 1 个 block 的 [CLS] 注意力图（8×8 patch）。

运行（项目根目录）：
    python applications/donkeycar_vit.py            # 完整（约 6~10 分钟）
    python applications/donkeycar_vit.py --quick    # 快速自检（约 1 分钟）

产物（outputs/donkeycar_vit/）：
    测试报告.md、训练曲线.png、开环误差分布.png、模型对比.png、
    鲁棒性对比.png、多赛道闭环.png、注意力图.png、donkeycar_vit_结果.json
"""

# ==================== 标准脚本初始化头（契约第 1 节，原样复制） ====================
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common_setup import init_env
init_env(seed=42)
# ==============================================================================

import argparse
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from miniflows.graph import Node
from miniflows.losses import MSELoss
from miniflows.optim import Adam
from miniflows.trainer import Trainer
from miniflows.transformer import build_vit
from miniflows.utils import save_model, load_model, count_params

# 复用 CNN 评测脚本的数据/指标/闭环工具，保证协议完全一致
from applications.donkeycar import generate_track_dataset, _render_from_row_offsets
from applications.donkeycar import IMG_SIZE, ROAD_HALF_W
from applications.donkeycar_eval import (split_indices, regression_metrics,
                                         predict_array, perturb_images,
                                         ROBUSTNESS_KINDS, closed_loop_run,
                                         sample_tracks, track_speed,
                                         make_predict_fn)

OUT_DIR = os.path.join(_ROOT, 'outputs', 'donkeycar_vit')
os.makedirs(OUT_DIR, exist_ok=True)
CNN_JSON = os.path.join(_ROOT, 'outputs', 'donkeycar', 'donkeycar_测试结果.json')

# ------------------------------ ViT 配置 ------------------------------
VIT_CFG = dict(img_size=IMG_SIZE, patch_size=8, dim=48, depth=2,
               heads=4, mlp_ratio=2)
DEFAULT_SEEDS = 5
DEFAULT_SAMPLES = 900
DEFAULT_EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-3
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


# ======================================================================
# 一、训练与开环评估
# ======================================================================
def train_one_seed(seed, X, y, epochs):
    """按与 CNN 评测完全相同的划分/种子训练一个 ViT，并在留出测试集上评估。"""
    idx_tr, idx_va, idx_te = split_indices(len(X), seed)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]
    X_te, y_te = X[idx_te], y[idx_te]

    np.random.seed(seed)                                 # 固定参数初始化
    model = build_vit(**VIT_CFG)
    optimizer = Adam(model.params(), lr=LR)
    trainer = Trainer(model, MSELoss(), optimizer, task='regression', seed=seed)

    t0 = time.perf_counter()
    hist = trainer.train(X_tr, y_tr, X_val=X_va, y_val=y_va,
                         epochs=epochs, batch_size=BATCH_SIZE, verbose=False)
    train_seconds = time.perf_counter() - t0

    pred = predict_array(model, X_te)
    metrics = regression_metrics(pred, y_te)
    baseline = regression_metrics(np.zeros_like(y_te), y_te)
    return dict(seed=seed, model=model, history=hist, metrics=metrics,
                baseline=baseline, pred=pred, y_test=y_te, X_test=X_te,
                train_seconds=train_seconds,
                split_sizes=(len(idx_tr), len(idx_va), len(idx_te)))


# ======================================================================
# 二、鲁棒性
# ======================================================================
def robustness_suite(model, X_test, y_test, seed):
    """对测试集施加 11 种扰动，返回每种扰动的 MAE。"""
    rng = np.random.RandomState(1000 + seed)
    out = {}
    for kind in ROBUSTNESS_KINDS:
        Xp = perturb_images(X_test, kind, rng)
        pred = predict_array(model, Xp)
        out[kind] = regression_metrics(pred, y_test)['mae']
    return out


# ======================================================================
# 三、绘图
# ======================================================================
def plot_training(histories, path):
    """5 个种子的训练/验证损失曲线（细线）+ 均值（粗线）。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for h in histories:
        ax.plot(h['train_loss'], color='#1f77b4', alpha=0.2, lw=1)
        ax.plot(h['val_loss'], color='#d62728', alpha=0.2, lw=1)
    n = min(len(h['train_loss']) for h in histories)
    tr = np.array([h['train_loss'][:n] for h in histories])
    va = np.array([h['val_loss'][:n] for h in histories])
    ax.plot(tr.mean(0), color='#1f77b4', lw=2, label='训练损失（均值）')
    ax.plot(va.mean(0), color='#d62728', lw=2, label='验证损失（均值）')
    ax.set_xlabel('轮次 Epoch'); ax.set_ylabel('MSE 损失')
    ax.set_title(f'DonkeyCar ViT：{len(histories)} 个随机种子的训练曲线')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_error_hist(all_err, path):
    """ViT 在全部留出测试样本上的绝对误差分布。"""
    err = np.asarray(all_err)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(err, bins=40, color='#9467bd', alpha=0.75, edgecolor='white')
    ax.axvline(err.mean(), color='#d62728', lw=2, label=f'MAE={err.mean():.4f}')
    ax.axvline(np.percentile(err, 95), color='#ff7f0e', ls='--', lw=2,
               label=f'P95={np.percentile(err, 95):.4f}')
    ax.set_xlabel('|预测转向角 - 真实转向角|'); ax.set_ylabel('样本数')
    ax.set_title(f'DonkeyCar ViT 开环测试：绝对误差分布（{len(err)} 条测试样本）')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_model_comparison(vit_open, cnn_open, vit_cl, cnn_cl, path):
    """ViT vs CNN：开环 MAE/P95 与闭环最大偏差/成功率 对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    names = ['ViT', 'CNN']

    ax = axes[0]
    vals = [vit_open['mae']['mean'], cnn_open['mae']['mean']]
    errs = [vit_open['mae']['std'], cnn_open['mae']['std']]
    ax.bar(names, vals, yerr=errs, color=['#9467bd', '#1f77b4'], alpha=0.85, capsize=4)
    ax.axhline(0.1, color='#d62728', ls='--', lw=1.2, label='目标 MAE=0.1')
    ax.set_ylabel('测试 MAE'); ax.set_title('开环：MAE（越低越好）')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=9)
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    vals = [vit_open['p95_abs']['mean'], cnn_open['p95_abs']['mean']]
    ax.bar(names, vals, color=['#9467bd', '#1f77b4'], alpha=0.85)
    ax.set_ylabel('P95 绝对误差'); ax.set_title('开环：P95 误差（长尾风险）')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    vals = [vit_cl['in_dist']['max_dev_mean'], cnn_cl['in_dist']['max_dev_mean']]
    ax.bar(names, vals, color=['#9467bd', '#1f77b4'], alpha=0.85)
    ax.axhline(ROAD_HALF_W, color='#d62728', ls='--', lw=1.2, label='道路半宽 10px')
    ax.set_ylabel('同分布闭环最大偏差均值 (px)')
    ax.set_title(f"闭环：成功 {vit_cl['in_dist']['success_rate']:.0%} vs "
                 f"{cnn_cl['in_dist']['success_rate']:.0%}")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f'{v:.2f}', ha='center', fontsize=9)
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    fig.suptitle('DonkeyCar：ViT 与 CNN 同协议对比（5 种子、同划分、同超参）', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path, dpi=150); plt.close(fig)


def plot_robustness(vit_agg, cnn_agg, path):
    """分组条形图：每种扰动下 ViT 与 CNN 的 MAE。"""
    kinds = ROBUSTNESS_KINDS
    x = np.arange(len(kinds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, [vit_agg[k]['mean'] for k in kinds], w,
           yerr=[vit_agg[k]['std'] for k in kinds], label='ViT',
           color='#9467bd', alpha=0.85, capsize=2)
    ax.bar(x + w / 2, [cnn_agg[k]['mean'] for k in kinds], w,
           yerr=[cnn_agg[k]['std'] for k in kinds], label='CNN',
           color='#1f77b4', alpha=0.85, capsize=2)
    ax.axhline(0.1, color='#d62728', ls='--', lw=1.2, label='目标 MAE=0.1')
    ax.set_xticks(x); ax.set_xticklabels(kinds, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('测试 MAE'); ax.set_title('DonkeyCar 鲁棒性对比：ViT vs CNN（5 种子）')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_closed_loop(results, tracks, path):
    """ViT 多赛道闭环：2 同分布 + 2 分布外 + 2 极端。"""
    show = [t for t in tracks if t['kind'] == '同分布'][:2] + \
           [t for t in tracks if t['kind'] == '分布外'][:2] + \
           [t for t in tracks if t['kind'] == '极端'][:2]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, tr in zip(axes.ravel(), show):
        A, lam, phase = tr['A'], tr['lam'], tr['phase']
        spd = track_speed(tr)
        zmax = spd * 160 + 20
        zg = np.linspace(-5, zmax, 300)
        cg = A * np.sin(2.0 * np.pi * zg / lam + phase)
        ax.fill_betweenx(zg, cg - ROAD_HALF_W, cg + ROAD_HALF_W,
                         color='#bbbbbb', alpha=0.7, label='路面')
        ax.plot(cg, zg, 'k--', lw=1, label='中心线')
        op = results[(tr['id'], 'open')]
        ax.plot(op['xs'], op['zs'], 'b--', lw=1.2, label='不转向开环')
        closed = [r for (k, r) in results.items() if k[0] == tr['id'] and k[1] != 'open']
        for r in closed:
            ax.plot(r['xs'], r['zs'], 'r-', lw=0.8, alpha=0.35)
        L = min(len(r['xs']) for r in closed)
        ax.plot(np.mean([r['xs'][:L] for r in closed], axis=0),
                np.mean([r['zs'][:L] for r in closed], axis=0),
                'r-', lw=2.2, label='ViT 闭环（均值）')
        ok = sum(r['success'] for r in closed)
        ax.set_title(f"{tr['id']}（{tr['kind']}）A={A:.1f}px λ={lam:.0f}px\n"
                     f"成功 {ok}/{len(closed)}，最大偏差均值 "
                     f"{np.mean([r['max_dev'] for r in closed]):.2f}px", fontsize=9)
        ax.set_xlabel('横向 x（px）'); ax.set_ylabel('前进 z（px）')
        ax.set_xlim(-32, 32); ax.set_xticks([-20, 0, 20])
        ax.set_aspect(0.2); ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9)
    fig.suptitle('DonkeyCar ViT 多赛道闭环：模型转向 vs 不转向开环', fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]); fig.savefig(path, dpi=150); plt.close(fig)


def plot_attention(model, X_sel, y_true, y_pred, path):
    """可视化第 1 个 block 的 [CLS] 注意力：模型在"看"哪些 patch。

    做法：
        1. 对 4 张测试图前向一次（模型内部缓存 last_attn）；
        2. 取第 1 个 block 的注意力 (N,H,L+1,L+1)；
        3. 对头取平均，取 CLS 查询行（index 0），去掉 CLS 列 → L 个 patch 权重；
        4. reshape 成 8×8，叠加在原图上（jet 热力图，alpha=0.45）。
    """
    model.train_mode(False)
    _ = model(Node(X_sel))                                # 触发前向，缓存注意力
    attn = model.attention_maps(0)                        # (N,H,65,65)
    a = attn.mean(axis=1)[:, 0, 1:]                       # (N,L) 平均头 + CLS 行
    grid = int(np.sqrt(a.shape[1]))                       # L=64 -> 8
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for i in range(min(4, len(X_sel))):
        ax = axes[0, i]
        ax.imshow(X_sel[i, 0], cmap='gray')
        ax.set_title(f'输入（真值 {y_true[i]:+.2f}）', fontsize=9)
        ax.axis('off')

        ax = axes[1, i]
        ax.imshow(X_sel[i, 0], cmap='gray')
        w = a[i].reshape(grid, grid)
        w = (w - w.min()) / (w.max() - w.min() + 1e-9)    # 归一化到 [0,1]
        ax.imshow(w, cmap='jet', alpha=0.45, extent=(0, IMG_SIZE, IMG_SIZE, 0),
                  interpolation='bilinear')
        ax.set_title(f'[CLS] 注意力（预测 {y_pred[i]:+.2f}）', fontsize=9)
        ax.axis('off')
    fig.suptitle('DonkeyCar ViT：第 1 个 Transformer Block 的 [CLS] 注意力（8×8 patch）',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.5); fig.savefig(path, dpi=150); plt.close(fig)


# ======================================================================
# 四、主流程
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='DonkeyCar ViT 版本评测')
    parser.add_argument('--quick', action='store_true', help='快速自检（2 种子/6 轮）')
    parser.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    suffix = '_quick' if args.quick else ''
    seeds = list(range(2 if args.quick else args.seeds))
    epochs = 6 if args.quick else args.epochs
    n_ind, n_stress, n_extreme = (3, 2, 1) if args.quick else (6, 4, 2)

    print('=' * 70)
    print('DonkeyCar ViT 评测（与 CNN 同协议）')
    print(f'ViT 配置: {VIT_CFG}')
    print(f'样本 {args.samples} ｜ 70/15/15 ｜ 种子 {len(seeds)} 个 ｜ 每模型 {epochs} 轮')
    print('=' * 70)

    # ---------- 1. 数据 ----------
    X, y = generate_track_dataset(args.samples, seed=2026)
    print(f'数据：X={X.shape}，y∈[{y.min():.3f}, {y.max():.3f}]')

    # ---------- 2. 逐种子训练 + 开环 + 鲁棒性 ----------
    runs = []
    rob_per_seed = {k: [] for k in ROBUSTNESS_KINDS}
    all_abs_err = []
    for seed in seeds:
        print(f'\n[种子 {seed}] 训练 ViT ...')
        r = train_one_seed(seed, X, y, epochs)
        runs.append(r)
        m = r['metrics']
        print(f"  MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} P95={m['p95_abs']:.4f} "
              f"R²={m['r2']:.3f}（{r['train_seconds']:.1f}s，划分 {r['split_sizes']}）")
        all_abs_err.append(np.abs(r['pred'] - r['y_test']))
        rob = robustness_suite(r['model'], r['X_test'], r['y_test'], seed)
        for k, v in rob.items():
            rob_per_seed[k].append(v)

    keys = ['mse', 'rmse', 'mae', 'p95_abs', 'max_abs', 'r2']
    vit_open = {k: {'mean': float(np.mean([r['metrics'][k] for r in runs])),
                    'std': float(np.std([r['metrics'][k] for r in runs]))}
                for k in keys}
    vit_rob = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
               for k, v in rob_per_seed.items()}
    vit_train_s = float(np.mean([r['train_seconds'] for r in runs]))
    params = count_params(runs[0]['model'])

    print('\n' + '-' * 70)
    print(f"ViT {len(runs)} 种子：MAE={vit_open['mae']['mean']:.4f}±"
          f"{vit_open['mae']['std']:.4f}，R²={vit_open['r2']['mean']:.3f}，"
          f"参数量={params:,}，平均训练 {vit_train_s:.1f}s")

    # ---------- 3. 读取 CNN 基线 ----------
    cnn = None
    if os.path.exists(CNN_JSON):
        with open(CNN_JSON, encoding='utf-8') as f:
            cnn = json.load(f)
        print(f"已读取 CNN 基线：{CNN_JSON}")
    else:
        print(f'[提示] 未找到 CNN 基线 {CNN_JSON}，请先运行 '
              f'python applications/donkeycar_eval.py；本报告只含 ViT 结果。')

    # ---------- 4. 闭环 ----------
    tracks = sample_tracks(n_ind, n_stress, n_extreme)
    print(f'\n闭环仿真：{len(tracks)} 条赛道 × {len(runs)} 个 ViT ...')
    cl_results = {}
    cl_rows = []
    for tr in tracks:
        spd = track_speed(tr)
        cl_results[(tr['id'], 'open')] = closed_loop_run(None, tr, speed=spd)
        for mi, r in enumerate(runs):
            fn = make_predict_fn(r['model'])
            res = closed_loop_run(fn, tr, speed=spd)
            cl_results[(tr['id'], mi)] = res
            cl_rows.append({'track': tr['id'], 'kind': tr['kind'], 'model': mi,
                            'success': res['success'], 'max_dev': res['max_dev'],
                            'mean_dev': res['mean_dev'], 'steps': res['steps'],
                            'mean_abs_delta': res['mean_abs_delta'],
                            'saturate_ratio': res['saturate_ratio']})
        ok = sum(cl_results[(tr['id'], mi)]['success'] for mi in range(len(runs)))
        md = np.mean([cl_results[(tr['id'], mi)]['max_dev'] for mi in range(len(runs))])
        print(f"  {tr['id']:>5s} {tr['kind']} A={tr['A']:.1f} λ={tr['lam']:.0f} "
              f"成功 {ok}/{len(runs)} 最大偏差均值 {md:.2f}px")

    # 噪声闭环（同分布 6 条 + σ=0.05）
    noise_rows = []
    for tr in [t for t in tracks if t['kind'] == '同分布']:
        for mi, r in enumerate(runs):
            fn = make_predict_fn(r['model'])
            rng = np.random.RandomState(5000 + mi)
            res = closed_loop_run(fn, tr, noise_sigma=0.05, rng=rng,
                                  speed=track_speed(tr))
            noise_rows.append({'track': tr['id'], 'model': mi,
                               'success': res['success'], 'max_dev': res['max_dev'],
                               'mean_dev': res['mean_dev'], 'steps': res['steps'],
                               'mean_abs_delta': res['mean_abs_delta'],
                               'saturate_ratio': res['saturate_ratio']})

    def summarise(rows):
        return {
            'n': len(rows),
            'success_rate': float(np.mean([r['success'] for r in rows])),
            'max_dev_mean': float(np.mean([r['max_dev'] for r in rows])),
            'max_dev_p95': float(np.percentile([r['max_dev'] for r in rows], 95)),
            'mean_dev_mean': float(np.mean([r['mean_dev'] for r in rows])),
            'mean_steps': float(np.mean([r['steps'] for r in rows])),
        }

    open_rows = []
    for tr in tracks:
        op = cl_results[(tr['id'], 'open')]
        open_rows.append({'success': op['success'], 'max_dev': op['max_dev'],
                          'mean_dev': op['mean_dev'], 'steps': op['steps']})
    vit_cl = {
        'in_dist': summarise([r for r in cl_rows if r['kind'] == '同分布']),
        'out_dist': summarise([r for r in cl_rows if r['kind'] == '分布外']),
        'extreme': summarise([r for r in cl_rows if r['kind'] == '极端']),
        'noise_0.05': summarise(noise_rows),
        'open_loop': summarise(open_rows),
    }
    print(f"  同分布 {vit_cl['in_dist']['success_rate']:.0%}，"
          f"分布外 {vit_cl['out_dist']['success_rate']:.0%}，"
          f"极端 {vit_cl['extreme']['success_rate']:.0%}，"
          f"噪声 {vit_cl['noise_0.05']['success_rate']:.0%}，"
          f"开环 {vit_cl['open_loop']['success_rate']:.0%}")

    # ---------- 5. 持久化 / 延迟 / 边界 ----------
    model0 = runs[0]['model']
    npz = os.path.join(OUT_DIR, f'donkeycar_vit_model{suffix}.npz')
    save_model(model0, npz)
    fresh = build_vit(**VIT_CFG)
    load_model(fresh, npz)
    diff = float(np.max(np.abs(predict_array(model0, runs[0]['X_test'][:32])
                               - predict_array(fresh, runs[0]['X_test'][:32]))))

    xin = runs[0]['X_test'][0, 0]
    for _ in range(3):
        _ = model0(Node(xin[None, None]))
    lat = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model0(Node(xin[None, None]))
        lat.append(time.perf_counter() - t0)
    lat_ms = float(np.mean(lat) * 1000)
    fps = 1000.0 / lat_ms

    fn0 = make_predict_fn(model0)
    edge_cases = {
        '全黑': np.zeros((IMG_SIZE, IMG_SIZE), np.float32),
        '全白': np.ones((IMG_SIZE, IMG_SIZE), np.float32),
        '纯噪声': np.random.RandomState(0).rand(IMG_SIZE, IMG_SIZE).astype(np.float32),
        '道路偏出画面': np.clip(_render_from_row_offsets(np.full(IMG_SIZE, 60.0)) / 255.0,
                          0.0, 1.0).astype(np.float32),
    }
    edge = {k: float(fn0(v)) for k, v in edge_cases.items()}

    # ---------- 6. 注意力图 ----------
    y_te = runs[0]['y_test']
    pred0 = runs[0]['pred']
    order = np.argsort(y_te)
    idx_sel = np.array([order[0], order[len(order) // 3],
                        order[2 * len(order) // 3], order[-1]])
    plot_attention(model0, runs[0]['X_test'][idx_sel], y_te[idx_sel],
                   pred0[idx_sel], os.path.join(OUT_DIR, f'注意力图{suffix}.png'))

    # ---------- 7. 图 ----------
    plot_training([r['history'] for r in runs],
                  os.path.join(OUT_DIR, f'训练曲线{suffix}.png'))
    plot_error_hist(np.concatenate(all_abs_err),
                    os.path.join(OUT_DIR, f'开环误差分布{suffix}.png'))
    plot_closed_loop(cl_results, tracks, os.path.join(OUT_DIR, f'多赛道闭环{suffix}.png'))
    if cnn is not None:
        plot_model_comparison(vit_open, cnn['open_loop']['mean_std'],
                              vit_cl, cnn['closed_loop']['summary'],
                              os.path.join(OUT_DIR, f'模型对比{suffix}.png'))
        plot_robustness(vit_rob, cnn['robustness']['per_kind'],
                        os.path.join(OUT_DIR, f'鲁棒性对比{suffix}.png'))

    # ---------- 8. JSON ----------
    result = {
        'model': 'vit', 'config': VIT_CFG, 'params': params,
        'protocol': {'samples': args.samples, 'seeds': seeds, 'epochs': epochs,
                     'batch_size': BATCH_SIZE, 'lr': LR,
                     'split': [TRAIN_FRAC, VAL_FRAC, 1 - TRAIN_FRAC - VAL_FRAC],
                     'test_used_for_validation': False},
        'open_loop': {'per_seed': [
            {'seed': r['seed'], **r['metrics'],
             'train_seconds': r['train_seconds']} for r in runs],
            'mean_std': vit_open},
        'robustness': vit_rob,
        'closed_loop': vit_cl,
        'persistence_max_diff': diff,
        'latency_ms': lat_ms, 'fps': fps,
        'edge_cases': edge,
        'train_seconds_mean': vit_train_s,
    }
    json_path = os.path.join(OUT_DIR, f'donkeycar_vit_结果{suffix}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)

    write_report(result, vit_open, vit_rob, vit_cl, cnn, tracks, cl_rows, suffix)
    print(f'\n完成：\n  报告 {os.path.join(OUT_DIR, f"测试报告{suffix}.md")}\n'
          f'  JSON {json_path}\n  图片 6 张（训练/误差/对比/鲁棒性/闭环/注意力）')


# ======================================================================
# 五、报告
# ======================================================================
def write_report(result, vit_open, vit_rob, vit_cl, cnn, tracks, cl_rows, suffix=''):
    """生成中文 Markdown 报告，含 ViT 与 CNN 的同协议对比。"""
    L = []
    L.append('# DonkeyCar ViT 版本测试报告\n')
    L.append('> 由 `applications/donkeycar_vit.py` 自动生成。'
             '协议与 `donkeycar_eval.py` 完全一致：900 样本、70/15/15、'
             '测试集不参与训练/早停/选模型、5 个随机种子、12 轮。\n')

    L.append('## 一、模型与协议\n')
    c = result['config']
    L.append(f"- ViT-Tiny：patch={c['patch_size']}，dim={c['dim']}，"
             f"depth={c['depth']}，heads={c['heads']}，mlp_ratio={c['mlp_ratio']}\n"
             f"- 参数量：**{result['params']:,}**（CNN 版 44,426，基本一致）\n"
             f"- 训练：Adam lr={result['protocol']['lr']}，batch={result['protocol']['batch_size']}，"
             f"{result['protocol']['epochs']} 轮，5 个种子\n")

    L.append('## 二、开环测试（留出 15% 测试集）\n')
    L.append('| 指标 | ViT（均值±标准差） | CNN（均值±标准差） | 结论 |')
    L.append('|---|---|---|---|')
    cnn_open = cnn['open_loop']['mean_std'] if cnn else None
    rows = [('MAE', 'mae'), ('RMSE', 'rmse'), ('P95', 'p95_abs'), ('R²', 'r2')]
    for label, key in rows:
        v = vit_open[key]
        if cnn_open:
            cv = cnn_open[key]
            better = 'ViT 更好' if ((key == 'r2' and v['mean'] > cv['mean']) or
                                    (key != 'r2' and v['mean'] < cv['mean'])) else 'CNN 更好'
            L.append(f"| {label} | {v['mean']:.4f}±{v['std']:.4f} | "
                     f"{cv['mean']:.4f}±{cv['std']:.4f} | {better} |")
        else:
            L.append(f"| {label} | {v['mean']:.4f}±{v['std']:.4f} | — | — |")
    L.append('')
    L.append(f"**结论**：ViT 在留出测试集上 MAE = **{vit_open['mae']['mean']:.4f}**，"
             f"R² = {vit_open['r2']['mean']:.3f}；每个模型平均训练 "
             f"{result['train_seconds_mean']:.1f}s。\n")

    L.append('## 三、鲁棒性（开环 MAE，5 种子）\n')
    L.append('| 扰动 | ViT | CNN | 差异 |')
    L.append('|---|---|---|---|')
    cnn_rob = cnn['robustness']['per_kind'] if cnn else {}
    for k in ROBUSTNESS_KINDS:
        v = vit_rob[k]
        if cnn_rob:
            cv = cnn_rob[k]
            d = (v['mean'] - cv['mean']) / max(cv['mean'], 1e-9) * 100
            L.append(f"| {k} | {v['mean']:.4f}±{v['std']:.4f} | "
                     f"{cv['mean']:.4f}±{cv['std']:.4f} | {d:+.1f}% |")
        else:
            L.append(f"| {k} | {v['mean']:.4f}±{v['std']:.4f} | — | — |")
    L.append('')

    L.append('## 四、闭环仿真\n')
    L.append('| 场景 | ViT 成功率 | ViT 最大偏差 | CNN 成功率 | CNN 最大偏差 |')
    L.append('|---|---|---|---|---|')
    cnn_cl = cnn['closed_loop']['summary'] if cnn else {}
    names = [('in_dist', '同分布'), ('out_dist', '分布外'), ('extreme', '极端'),
             ('noise_0.05', '同分布+噪声σ=0.05'), ('open_loop', '不转向开环')]
    for key, label in names:
        v = vit_cl[key]
        if cnn_cl and key in cnn_cl:
            cv = cnn_cl[key]
            L.append(f"| {label} | {v['success_rate']:.0%} | {v['max_dev_mean']:.2f}px | "
                     f"{cv['success_rate']:.0%} | {cv['max_dev_mean']:.2f}px |")
        else:
            L.append(f"| {label} | {v['success_rate']:.0%} | {v['max_dev_mean']:.2f}px | — | — |")
    L.append('')
    L.append('> 成功判据：全程 |横向偏差| ≤ 10px。极端赛道 A=18~22px、'
             'λ=160~190px、速度 3.0，用于寻找失效边界。\n')

    L.append('## 五、持久化、延迟与边界\n')
    L.append(f"- 保存/加载预测最大差：**{result['persistence_max_diff']:.2e}**\n"
             f"- 单帧推理：**{result['latency_ms']:.2f} ms**（≈{result['fps']:.0f} FPS，"
             f"CPU，batch=1，float64）\n")
    L.append('| 边界输入 | 输出 | 在 [-1,1] |')
    L.append('|---|---|---|')
    for k, v in result['edge_cases'].items():
        L.append(f"| {k} | {v:+.4f} | {bool(-1 <= v <= 1)} |")
    L.append('')

    L.append('## 六、注意力可视化\n')
    L.append('![注意力图](注意力图%s.png)\n' % suffix)
    L.append('> 取第 1 个 Transformer Block 的 [CLS] 查询注意力，对 4 个头取平均，'
             '得到 8×8 patch 权重。热区表示模型做转向决策时主要关注的图像区域。\n')

    L.append('## 七、总体结论\n')
    if cnn_open:
        L.append(f"1. 同协议下 ViT MAE **{vit_open['mae']['mean']:.4f}**，"
                 f"CNN **{cnn_open['mae']['mean']:.4f}**；参数量分别为 "
                 f"{result['params']:,} 与 44,426，基本对齐，比较公平；\n")
    else:
        L.append(f"1. ViT MAE **{vit_open['mae']['mean']:.4f}**，R² "
                 f"{vit_open['r2']['mean']:.3f}；\n")
    L.append(f"2. ViT 同分布闭环成功率 **{vit_cl['in_dist']['success_rate']:.0%}**、"
             f"分布外 **{vit_cl['out_dist']['success_rate']:.0%}**、"
             f"极端 **{vit_cl['extreme']['success_rate']:.0%}**；\n")
    L.append(f"3. 鲁棒性：ViT 在干净集 MAE {vit_rob['clean']['mean']:.4f}；"
             f"最差扰动为「{max(ROBUSTNESS_KINDS, key=lambda k: vit_rob[k]['mean'])}」"
             f"（{max(vit_rob[k]['mean'] for k in ROBUSTNESS_KINDS):.4f}）；\n")
    L.append(f"4. 单帧 {result['latency_ms']:.2f} ms（≈{result['fps']:.0f} FPS），"
             f"训练每模型约 {result['train_seconds_mean']:.1f}s；\n")
    L.append('5. **局限**：仍是合成赛道与简化运动学；ViT 的 patch 级注意力只有 8×8，'
             '不是像素级定位；结论不能外推到真车。\n')

    # ---------- 补充分析：统计口径 + 鲁棒性短板 ----------
    L.append('## 八、统计口径与补充分析\n')
    if cnn_open:
        diff = abs(vit_open['mae']['mean'] - cnn_open['mae']['mean'])
        sd = max(vit_open['mae']['std'], cnn_open['mae']['std'])
        L.append(f"- **开环差异是否显著**：ViT 与 CNN 的 MAE 差 {diff:.4f}，"
                 f"两者标准差的最大值为 {sd:.4f}；差值 {'小于' if diff < sd else '不小于'} 标准差，"
                 f"{'不能宣称 ViT 优于 CNN，应视为同一水平' if diff < sd else '差异可能具有统计意义，但仍需多种子检验'}。\n")
    if cnn_rob:
        L.append(f"- **鲁棒性短板**：ViT 在「底部遮挡」上的 MAE 为 "
                 f"{vit_rob['底部遮挡']['mean']:.4f}，CNN 为 "
                 f"{cnn_rob['底部遮挡']['mean']:.4f}（约 "
                 f"{vit_rob['底部遮挡']['mean'] / max(cnn_rob['底部遮挡']['mean'], 1e-9):.1f} 倍）；"
                 f"「亮度-0.2」为 {vit_rob['亮度-0.2']['mean']:.4f} vs "
                 f"{cnn_rob['亮度-0.2']['mean']:.4f}。"
                 f"原因：ViT 没有卷积的局部性/平移等变先验，且训练样本仅 630 条；"
                 f"全局亮度/对比度变化会同时改变所有 patch 的表示。\n")
        L.append('- **可改进方向**：训练时加入亮度/对比度/遮挡增强；使用更大数据集；'
                 '或用卷积 stem（Hybrid ViT）替代线性 patch embedding。\n')

    with open(os.path.join(OUT_DIR, f'测试报告{suffix}.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    main()
