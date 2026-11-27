# -*- coding: utf-8 -*-
"""在 DonkeySim 中采集大偏移纠偏（Recovery）与 DAgger 闭环纠正数据。

解决纯贴中线 Behavioral Cloning 的分布偏移问题：
1. recovery 模式：从赛道不同节点以较大横向偏移（±0.28）和偏航角（±20°）出发，
   由专家 Route.follow() 实时拉回中线并记录 (image, expert_steering)；
2. dagger 模式：由当前模型控制车辆行驶，每帧由专家 Route.follow() 标注正确纠偏转向，
   专门覆盖模型在真实闭环中容易漂移进入的状态分布。
"""

import argparse
import base64
import csv
import io
import json
import math
from pathlib import Path
import socket
import sys
from uuid import uuid4

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.collect_donkeysim_rgb import (
    Route,
    Simulator,
    WORLD_SCALE,
    restart_at_node,
    save_frame,
)
from applications.donkeycar_rgb import prepare_frame
from applications.donkeysim_rgb_runner import (
    DEFAULT_WEIGHTS,
    _connect_and_enter_scene,
    load_rgb_steering_model,
)
from miniflows.graph import constant


def run_recovery_collection(
    output_dir,
    scene='current',
    port=9091,
    total_frames=2400,
    dagger_ratio=0.5,
    throttle=0.10,
    steps_per_ep=240,
    every=2,
    weights=DEFAULT_WEIGHTS,
    seed=2026,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / 'images'
    images_dir.mkdir(exist_ok=True)
    manifest_path = out / 'samples.csv'

    model = None
    if dagger_ratio > 0:
        model, _ = load_rgb_steering_model(weights)

    fields = ('image', 'scene', 'episode', 'frame', 'mode', 'steering', 'model_steering',
              'throttle', 'speed', 'cte', 'route_err')
    saved = 0
    ep_idx = 0
    rng = np.random.default_rng(seed)

    sock, sim, initial = _connect_and_enter_scene('127.0.0.1', int(port), scene)
    with sock:
        try:
            node_count = int(initial['totalNodes'])
            route = Route(sim.read_route(node_count))
            spawn_pt = sim.route_node(int(initial['activeNode']))
            height = float(np.clip(float(initial['pos_y']) * WORLD_SCALE - spawn_pt[1], 0.33, 1.4))
            print(f'已连接赛道（{node_count} 节点，height={height:.2f}），目标采集 {total_frames} 帧', flush=True)

            with manifest_path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()

                while saved < total_frames:
                    ep_idx += 1
                    use_dagger = (model is not None) and (ep_idx % 2 == 0 if dagger_ratio >= 0.4 else rng.random() < dagger_ratio)
                    mode = 'dagger' if use_dagger else 'recovery'
                    node = int(rng.integers(4, max(5, node_count - 6)))
                    if mode == 'recovery':
                        # 较大横向偏移与车头偏角，专门训练“从路边拉回中线”
                        offset = float(rng.uniform(-0.32, 0.32))
                        yaw_off = float(rng.uniform(-22.0, 22.0))
                    else:
                        # DAgger 从轻中度偏移出发，让模型自己开暴露漂移状态
                        offset = float(rng.uniform(-0.18, 0.18))
                        yaw_off = float(rng.uniform(-10.0, 10.0))

                    try:
                        pkt = restart_at_node(sim, node, height, offset, yaw_off)
                    except ValueError as exc:
                        print(f'ep {ep_idx} 起点 {node} 跳过：{exc}', flush=True)
                        continue

                    ep_name = f'{scene}-{mode}-{ep_idx:04d}-{uuid4().hex[:6]}'
                    ep_saved = 0
                    reason = 'steps_limit'
                    for frame in range(steps_per_ep):
                        pos = np.array([float(pkt['pos_x']), float(pkt['pos_z'])])
                        yaw, speed, cte = float(pkt['yaw']), float(pkt['speed']), float(pkt['cte'])
                        hit = str(pkt.get('hit', 'none'))
                        if hit != 'none':
                            reason = f'collision:{hit}'
                            break
                        exp_steer, r_err = route.follow(pos, yaw, speed)
                        if exp_steer is None:
                            reason = 'end_of_route'
                            break
                        if r_err > 0.90:
                            reason = 'far_off_route'
                            break

                        m_steer = exp_steer
                        if mode == 'dagger':
                            raw = base64.b64decode(pkt['image'])
                            with Image.open(io.BytesIO(raw)) as im:
                                m_steer = float(model(constant(prepare_frame(im)[None, ...])).data.item())
                            # 若偏离较大(>0.45)，以 50% 概率混合专家拉回避免立刻出界，同时仍用专家标注
                            ctrl_steer = m_steer if r_err <= 0.45 else float(0.5 * m_steer + 0.5 * exp_steer)
                        else:
                            # recovery 模式下加极小探索噪声避免过于单一
                            ctrl_steer = float(np.clip(exp_steer + rng.normal(0.0, 0.03), -1.0, 1.0))

                        sim.control(ctrl_steer, throttle, 0.0)

                        # 对大偏差/大转向帧每步都存，中线小转向按 every 抽样，提高纠偏样本占比
                        should_save = (speed >= 0.05) and (
                            (r_err >= 0.12 or abs(exp_steer) >= 0.25) or (frame % every == 0)
                        )
                        if should_save:
                            rel_img = Path('images') / f'{ep_idx:04d}_{frame:04d}_{exp_steer:+.4f}.jpg'
                            size = save_frame(out / rel_img, pkt)
                            if size is not None:
                                writer.writerow({
                                    'image': rel_img.as_posix(),
                                    'scene': scene,
                                    'episode': ep_name,
                                    'frame': frame,
                                    'mode': mode,
                                    'steering': round(float(exp_steer), 6),
                                    'model_steering': round(float(m_steer), 6),
                                    'throttle': throttle,
                                    'speed': round(speed, 4),
                                    'cte': round(cte, 4),
                                    'route_err': round(r_err, 4),
                                })
                                saved += 1
                                ep_saved += 1
                                if saved >= total_frames:
                                    reason = 'frames_limit'
                                    break
                        if frame + 1 < steps_per_ep:
                            pkt = sim.receive('telemetry', timeout=5.0)
                    print(f'ep {ep_idx:02d} [{mode:8s}] node={node:4d} off={offset:+.2f} yaw={yaw_off:+5.1f}° -> +{ep_saved:3d} 帧 ({saved}/{total_frames}) [{reason}]', flush=True)
        finally:
            try:
                sim.control(0.0, 0.0, 1.0)
            except OSError:
                pass

    summary = {'scene': scene, 'total_frames': saved, 'episodes': ep_idx, 'manifest': str(manifest_path)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('采集完成：', json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scene', default='current')
    parser.add_argument('--port', type=int, default=9091)
    parser.add_argument('--total-frames', type=int, default=2400)
    parser.add_argument('--dagger-ratio', type=float, default=0.5)
    parser.add_argument('--throttle', type=float, default=0.10)
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--every', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    run_recovery_collection(
        output_dir=args.output_dir,
        scene=args.scene,
        port=args.port,
        total_frames=args.total_frames,
        dagger_ratio=args.dagger_ratio,
        throttle=args.throttle,
        steps_per_ep=args.steps,
        every=args.every,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
