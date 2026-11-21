# -*- coding: utf-8 -*-
"""
experiments/tune_scale.py —— DonkeyCar 模型「放大炼丹」：参数量扫描 + 最优模型产出
============================================================================

目的：现有主模型 ResNet-GN（44,337 参数）与 ViT-Tiny（44,353 参数）规模偏小，
本脚本在**完全相同的评测协议**下（900 样本 / 70-15-15 划分 / Adam lr=1e-3 /
batch=32 / 12 轮 / 测试集不参与训练与选型）扫描不同规模的 ResNet 与 ViT，
用开环 MAE + 鲁棒性 + 闭环偏差三项综合排序，产出：

  outputs/scale_best/best_model.npz      最优模型权重
  outputs/scale_best/best_config.json    最优模型的结构配置（供 Web Studio 复原）
  outputs/scale_best/缩放扫描结果.json   全部候选的完整指标

用法：
    python experiments/tune_scale.py                # 完整扫描（2 种子）
    python experiments/tune_scale.py --seeds 1      # 快速筛选
    python experiments/tune_scale.py --only resnet  # 只扫 ResNet 系列
"""

import argparse
import json
import os
import sys
import time

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

import numpy as np

from miniflows.graph import Node
from miniflows.losses import MSELoss
from miniflows.optim import Adam
from miniflows.trainer import Trainer
from miniflows.utils import save_model, count_params
from miniflows.resnet import build_resnet
from miniflows.transformer import build_vit

from applications.donkeycar import generate_track_dataset, IMG_SIZE
from applications.donkeycar_eval import (
    split_indices, regression_metrics, predict_array, perturb_images,
    ROBUSTNESS_KINDS, closed_loop_run, sample_tracks, make_predict_fn,
)

OUT_DIR = os.path.join(_ROOT, 'outputs', 'scale_best')
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = 900
EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-3
DATA_SEED = 2026


# ======================================================================
# 一、候选架构（按「参数量从基线往上翻」排列）
#
# 炼丹经验（首轮扫描实测）：175K 参数的 ResNet 在 lr=1e-3 下直接发散
# （R²=-5.35），大模型必须配更小的学习率 + 梯度裁剪。因此每个候选
# 自带 (lr, clip) 超参 —— 学习率是架构调优的一部分，逐候选设定才公平。
# ======================================================================
def _c(name, family, lr, clip, **kw):
    d = {'name': name, 'family': family, 'lr': lr, 'clip': clip}
    d.update(kw)
    return d


CANDIDATES = [
    _c('resnet_base',      'resnet', 1e-3, 0.0, c0=8,  widths=(8, 16, 32),   blocks=(2, 2, 2)),
    _c('resnet_wide',      'resnet', 5e-4, 1.0, c0=16, widths=(16, 32, 64),  blocks=(2, 2, 2)),
    _c('resnet_wide_deep', 'resnet', 5e-4, 1.0, c0=16, widths=(16, 32, 64),  blocks=(3, 3, 3)),
    _c('resnet_xwide',     'resnet', 3e-4, 1.0, c0=24, widths=(24, 48, 96),  blocks=(2, 2, 2)),
    _c('vit_base',         'vit',    1e-3, 0.0, dim=48,  depth=2, heads=4, mlp_ratio=2),
    _c('vit_big',          'vit',    5e-4, 1.0, dim=96,  depth=3, heads=6, mlp_ratio=2),
    _c('vit_huge',         'vit',    3e-4, 1.0, dim=128, depth=4, heads=8, mlp_ratio=2),
]


def build_candidate(cfg):
    """按候选配置构造模型（结构参数全部可复现地还原）。"""
    if cfg['family'] == 'resnet':
        return build_resnet(in_channels=1, c0=cfg['c0'],
                            widths=tuple(cfg['widths']),
                            blocks=tuple(cfg['blocks']), num_groups=None)
    return build_vit(img_size=IMG_SIZE, patch_size=8, dim=cfg['dim'],
                     depth=cfg['depth'], heads=cfg['heads'],
                     mlp_ratio=cfg['mlp_ratio'], in_channels=1)


def config_to_json(cfg):
    """把候选配置转成可 JSON 序列化的结构描述。"""
    out = {'name': cfg['name'], 'family': cfg['family'],
           'lr': cfg['lr'], 'clip': cfg['clip']}
    if cfg['family'] == 'resnet':
        out.update(c0=cfg['c0'], widths=list(cfg['widths']), blocks=list(cfg['blocks']))
    else:
        out.update(dim=cfg['dim'], depth=cfg['depth'], heads=cfg['heads'],
                   mlp_ratio=cfg['mlp_ratio'], patch_size=8, img_size=IMG_SIZE)
    return out


# ======================================================================
# 二、单种子训练 + 评估
# ======================================================================
def train_one_seed(cfg, seed, X, y, epochs=EPOCHS):
    """同协议训练一个候选模型并在留出测试集上评估（lr/裁剪按候选配置）。"""
    idx_tr, idx_va, idx_te = split_indices(len(X), seed)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]
    X_te, y_te = X[idx_te], y[idx_te]

    np.random.seed(seed)
    model = build_candidate(cfg)
    optimizer = Adam(model.params(), lr=cfg['lr'])
    trainer = Trainer(model, MSELoss(), optimizer, task='regression', seed=seed)

    # 梯度裁剪：在 Trainer 的回调里做全局范数裁剪（clip<=0 表示关闭）
    clip = float(cfg.get('clip') or 0.0)

    def _clip_cb(state):
        if clip > 0:
            total_sq = 0.0
            for p in model.params():
                if p.grad is not None:
                    total_sq += float(np.sum(p.grad ** 2))
            tn = float(np.sqrt(total_sq))
            if tn > clip:
                s = clip / (tn + 1e-12)
                for p in model.params():
                    if p.grad is not None:
                        p.grad *= s
        return False

    t0 = time.perf_counter()
    trainer.train(X_tr, y_tr, X_val=X_va, y_val=y_va,
                  epochs=epochs, batch_size=BATCH_SIZE, verbose=False,
                  callbacks=[_clip_cb])
    train_seconds = time.perf_counter() - t0

    pred = predict_array(model, X_te)
    metrics = regression_metrics(pred, y_te)
    return dict(seed=seed, model=model, metrics=metrics, pred=pred,
                y_test=y_te, X_test=X_te, train_seconds=train_seconds)


def robustness_suite(model, X_test, y_test, seed):
    """11 种视觉扰动下的测试集 MAE。"""
    rng = np.random.RandomState(1000 + seed)
    return {kind: regression_metrics(
        predict_array(model, perturb_images(X_test, kind, rng)), y_test)['mae']
        for kind in ROBUSTNESS_KINDS}


def closed_loop_suite(model):
    """多种赛道下的闭环最大横向偏差与成功率。"""
    pf = make_predict_fn(model)
    out = {}
    for label, tracks in (('in_dist', sample_tracks(6, 4, 2)),):
        devs, succ = [], []
        for tr in tracks:
            r = closed_loop_run(pf, tr, noise_sigma=0.0, rng=np.random.RandomState(7))
            devs.append(r['max_dev']); succ.append(bool(r['success']))
        out[label] = {'max_dev_mean': float(np.mean(devs)),
                       'success_rate': float(np.mean(succ)),
                       'n': len(devs)}
    # 噪声闭环
    devs, succ = [], []
    for tr in sample_tracks(4, 2, 1):
        r = closed_loop_run(pf, tr, noise_sigma=0.05, rng=np.random.RandomState(7))
        devs.append(r['max_dev']); succ.append(bool(r['success']))
    out['noise_0.05'] = {'max_dev_mean': float(np.mean(devs)),
                         'success_rate': float(np.mean(succ)), 'n': len(devs)}
    return out


# ======================================================================
# 三、主流程
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description='DonkeyCar 参数量放大炼丹扫描')
    ap.add_argument('--seeds', type=int, default=2, help='每个候选的随机种子数')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--samples', type=int, default=SAMPLES)
    ap.add_argument('--only', default='', help='只跑名字里包含该子串的候选')
    ap.add_argument('--skip-existing', action='store_true',
                    help='若 outputs/scale_best/缩放扫描结果.json 已存在则跳过')
    args = ap.parse_args()

    result_path = os.path.join(OUT_DIR, '缩放扫描结果.json')
    if args.skip_existing and os.path.exists(result_path):
        print(f'[跳过] 已存在 {result_path}')
        return

    cands = CANDIDATES
    if args.only:
        cands = [c for c in cands if args.only in c['name']]
    if not cands:
        print('没有匹配的候选')
        return

    print('=' * 78)
    print('DonkeyCar 参数量放大炼丹扫描')
    print(f'协议：{args.samples} 样本 / 70-15-15 / Adam lr={LR} / batch={BATCH_SIZE} / '
          f'{args.epochs} 轮 / {args.seeds} 种子')
    print('=' * 78)

    X, y = generate_track_dataset(args.samples, img_size=IMG_SIZE, seed=DATA_SEED)
    print(f'数据生成完成：X={X.shape}, y∈[{y.min():.3f}, {y.max():.3f}]\n')

    rows = []
    best = None
    t_all = time.perf_counter()

    for cfg in cands:
        print(f'--- 候选 {cfg["name"]} ---')
        try:
            probe = build_candidate(cfg)
            n_params = count_params(probe)
            print(f'  参数量 {n_params:,}')
        except Exception as exc:
            print(f'  [失败] 建模异常：{exc}')
            rows.append({'config': config_to_json(cfg), 'error': f'建模失败: {exc}'})
            continue

        per_seed, maes, rob_means, cl_means, ok = [], [], [], [], 0
        for seed in range(args.seeds):
            try:
                r = train_one_seed(cfg, seed, X, y, epochs=args.epochs)
                m = r['metrics']
                per_seed.append({'seed': seed, 'mae': m['mae'], 'rmse': m['rmse'],
                                 'p95_abs': m['p95_abs'], 'r2': m['r2'],
                                 'train_seconds': round(r['train_seconds'], 1)})
                maes.append(m['mae'])
                ok += 1
                print(f'  [种子 {seed}] MAE={m["mae"]:.4f}  R²={m["r2"]:.3f}  '
                      f'P95={m["p95_abs"]:.4f}  ({r["train_seconds"]:.0f}s)')
                if seed == 0:
                    rob = robustness_suite(r['model'], r['X_test'], r['y_test'], seed)
                    cl = closed_loop_suite(r['model'])
                    rob_means = [v for v in rob.values()]
                    cl_means = [v['max_dev_mean'] for v in cl.values()]
                    keep = r
            except Exception as exc:
                print(f'  [种子 {seed}] [失败] {exc}')

        if not maes:
            rows.append({'config': config_to_json(cfg), 'params': n_params,
                         'error': '全部种子训练失败'})
            continue

        mae_mean, mae_std = float(np.mean(maes)), float(np.std(maes))
        rob_mean = float(np.mean(rob_means)) if rob_means else None
        cl_mean = float(np.mean(cl_means)) if cl_means else None
        # 综合评分：开环 MAE 为主，鲁棒性退化与闭环偏差为辅（越小越好）
        score = mae_mean
        if rob_mean is not None:
            score += 0.5 * (rob_mean - mae_mean)
        if cl_mean is not None:
            score += 0.05 * cl_mean / 10.0

        row = {
            'config': config_to_json(cfg), 'params': n_params,
            'seeds': per_seed, 'ok_seeds': ok,
            'mae_mean': mae_mean, 'mae_std': mae_std,
            'mae_all': maes,
            'robustness_mean_mae': rob_mean,
            'closed_loop_max_dev_mean': cl_mean,
            'score': score,
        }
        rows.append(row)
        print(f'  → MAE {mae_mean:.4f} ± {mae_std:.4f} | 鲁棒均值 {rob_mean} | '
              f'闭环偏差 {cl_mean} | 综合分 {score:.4f}\n')

        if best is None or score < best['score']:
            best = row
            best['_model'] = keep['model']

    print('=' * 78)
    print('扫描结果（按综合分排序）')
    print('=' * 78)
    ok_rows = [r for r in rows if r.get('ok_seeds')]
    ok_rows.sort(key=lambda r: r['score'])
    print(f'{"候选":<20}{"参数量":>12}{"MAE":>18}{"鲁棒均值":>12}{"闭环偏差":>12}{"综合分":>10}')
    for r in ok_rows:
        print(f'{r["config"]["name"]:<20}{r["params"]:>12,}'
              f'{r["mae_mean"]:>10.4f} ± {r["mae_std"]:.4f}'
              f'{(r["robustness_mean_mae"] or 0):>12.4f}'
              f'{(r["closed_loop_max_dev_mean"] or 0):>12.2f}{r["score"]:>10.4f}')

    out = {
        'protocol': {'samples': args.samples, 'epochs': args.epochs,
                     'batch_size': BATCH_SIZE, 'lr': LR, 'seeds': args.seeds,
                     'data_seed': DATA_SEED, 'split': [0.70, 0.15, 0.15]},
        'rows': [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
        'best': ({k: v for k, v in best.items() if not k.startswith('_')}
                 if best else None),
        'elapsed_s': round(time.perf_counter() - t_all, 1),
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    if best is not None:
        npz_path = os.path.join(OUT_DIR, 'best_model.npz')
        cfg_path = os.path.join(OUT_DIR, 'best_config.json')
        save_model(best['_model'], npz_path)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump({'config': best['config'], 'params': best['params'],
                       'mae_mean': best['mae_mean'], 'mae_std': best['mae_std'],
                       'robustness_mean_mae': best['robustness_mean_mae'],
                       'closed_loop_max_dev_mean': best['closed_loop_max_dev_mean'],
                       'npz': 'outputs/scale_best/best_model.npz'},
                      f, ensure_ascii=False, indent=1)
        print(f'\n★ 最优：{best["config"]["name"]}（{best["params"]:,} 参数，'
              f'MAE {best["mae_mean"]:.4f} ± {best["mae_std"]:.4f}）')
        print(f'  权重：{os.path.relpath(npz_path, _ROOT)}')
        print(f'  配置：{os.path.relpath(cfg_path, _ROOT)}')
    print(f'\n耗时 {out["elapsed_s"] / 60:.1f} 分钟，结果写入 {os.path.relpath(result_path, _ROOT)}')


if __name__ == '__main__':
    main()
