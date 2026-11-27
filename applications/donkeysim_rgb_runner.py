# -*- coding: utf-8 -*-
"""DonkeySim 本机 RGB 实景闭环控制与实时遥测驱动。

复用本机 Unity 版 DonkeySim 的 TCP 协议（默认 127.0.0.1:9091），直接加载
applications/donkeycar_rgb.py 训练好的 RGB 转向模型执行真实闭环驾驶，
并可通过回调把原始 RGB 相机帧与车辆遥测实时推送给 Web 前端。

命令行独立测试示例：
    python applications/donkeysim_rgb_runner.py --scene generated_road --steps 600 --throttle 0.1
"""

import argparse
import base64
import io
import json
import math
from pathlib import Path
import socket
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.collect_donkeysim_rgb import Route, Simulator, WORLD_SCALE, restart_at_node
from applications.donkeycar_rgb import (
    CROP_TOP_FRACTION,
    HEIGHT as RGB_HEIGHT,
    WIDTH as RGB_WIDTH,
    build_model as build_rgb_model,
    prepare_frame as prepare_rgb_frame,
)
from miniflows.graph import constant
from miniflows.utils import load_model


DEFAULT_WEIGHTS = ROOT / 'outputs' / 'donkeycar_rgb' / 'three_scenes_steering' / 'steering_model.npz'
DEFAULT_SCENES = ('generated_road', 'mini_monaco', 'warehouse')


class _SocketLineReader:
    """基于 sock.recv 的安全按行读取器：单次超时不会像 sock.makefile() 那样永久损坏流。"""

    def __init__(self, sock):
        self.sock = sock
        self._buf = bytearray()

    def readline(self):
        while True:
            idx = self._buf.find(b'\n')
            if idx != -1:
                line = bytes(self._buf[:idx + 1])
                del self._buf[:idx + 1]
                return line
            chunk = self.sock.recv(65536)
            if not chunk:
                if self._buf:
                    line = bytes(self._buf)
                    self._buf.clear()
                    return line
                return b''
            self._buf.extend(chunk)


def load_rgb_steering_model(weights_path=None):
    """构建并加载 RGB 转向回归模型，同时校验权重键名。"""
    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(f'未找到 RGB 模型权重：{path}')
    model = build_rgb_model()
    with np.load(path, allow_pickle=False) as archive:
        expected = {name for name, _ in model.named_params()}
        if set(archive.files) != expected:
            raise ValueError(f'权重文件 {path.name} 与 donkeycar_rgb 模型结构不匹配')
    load_model(model, path)
    model.train_mode(False)
    return model, path


def _read_first_message(sim, timeout=3.0):
    """读取模拟器连接后的第一条 JSON 消息（可能是菜单就绪，也可能是赛道内遥测）。"""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('等待 DonkeySim 握手消息超时')
        sim.sock.settimeout(remaining)
        try:
            line = sim.stream.readline()
        except TimeoutError as exc:
            raise TimeoutError('等待 DonkeySim 握手消息超时') from exc
        if not line:
            raise ConnectionError('DonkeySim 已断开连接')
        return json.loads(line)


def probe_donkeysim(host='127.0.0.1', port=9091, timeout=2.5):
    """探测本机 DonkeySim 是否在线并读取可选场景列表。"""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sim = Simulator(sock, _SocketLineReader(sock))
            first = _read_first_message(sim, timeout=timeout)
            msg_type = first.get('msg_type')
            if msg_type == 'scene_selection_ready':
                sim.send('get_scene_names')
                scenes = sim.receive('scene_names', timeout=timeout).get('scene_names', list(DEFAULT_SCENES))
                return {
                    'connected': True,
                    'mode': 'menu',
                    'scenes': list(scenes),
                    'message': f'已连接 DonkeySim（{host}:{port}），停留在赛道选择菜单，发现 {len(scenes)} 个场景。',
                }
            if msg_type == 'telemetry':
                return {
                    'connected': True,
                    'mode': 'in_scene',
                    'scenes': list(DEFAULT_SCENES),
                    'message': f'已连接 DonkeySim（{host}:{port}），模拟器当前已在赛道内（启动闭环时会自动切场景或重置车辆）。',
                }
            return {
                'connected': True,
                'mode': str(msg_type or 'unknown'),
                'scenes': list(DEFAULT_SCENES),
                'message': f'已连接 DonkeySim（{host}:{port}），收到消息 {msg_type}。',
            }
    except (OSError, TimeoutError, ValueError, KeyError) as exc:
        return {
            'connected': False,
            'mode': 'offline',
            'scenes': list(DEFAULT_SCENES),
            'error': str(exc),
            'message': f'未检测到本机 DonkeySim（{host}:{port}）：{exc}。请先打开 Unity 版 DonkeySim 模拟器。',
        }


def _decode_rgb_and_jpeg(packet):
    """解码模拟器遥测包里的图像，返回 (PIL.Image RGB, jpeg_b64)。"""
    raw_b64 = packet.get('image')
    if not raw_b64:
        raise ValueError('遥测包缺少 image 字段')
    raw = base64.b64decode(raw_b64)
    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.format == 'JPEG' and img.mode == 'RGB':
        return img, raw_b64
    rgb = img.convert('RGB')
    buf = io.BytesIO()
    rgb.save(buf, format='JPEG', quality=85)
    return rgb, base64.b64encode(buf.getvalue()).decode('ascii')


def _connect_and_enter_scene(host, port, scene, timeout=15.0):
    """建立 TCP 连接并进入目标场景，返回 (sock, sim, initial_telemetry)。

    若模拟器当前已在另一赛道内且用户指定了新场景：先发 exit_scene，关闭旧连接，
    等待模拟器退回选赛道菜单后重新建连再 load_scene，避免旧赛道缓冲帧串包。
    """
    sock = socket.create_connection((host, int(port)), timeout=5.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sim = Simulator(sock, _SocketLineReader(sock))
    try:
        first = _read_first_message(sim, timeout=5.0)
        msg_type = first.get('msg_type')

        if msg_type in ('telemetry', 'car_loaded'):
            if not scene or scene == 'current':
                sim.control(0.0, 0.0, 1.0)
                sim.send('reset_car')
                pkt = sim.receive('telemetry', timeout=10.0)
                for _ in range(4):
                    pkt = sim.receive('telemetry', timeout=5.0)
                return sock, sim, pkt
            # 先退出当前场景并断开，等待模拟器退回菜单后重新连接
            sim.control(0.0, 0.0, 1.0)
            sim.send('exit_scene')
            sock.close()
            time.sleep(1.5)
            sock = socket.create_connection((host, int(port)), timeout=5.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sim = Simulator(sock, _SocketLineReader(sock))
            first = _read_first_message(sim, timeout=8.0)
            msg_type = first.get('msg_type')

        if msg_type != 'scene_selection_ready':
            sim.receive('scene_selection_ready', timeout=timeout)

        target_scene = scene if (scene and scene != 'current') else DEFAULT_SCENES[0]
        sim.send('get_scene_names')
        available = sim.receive('scene_names', timeout=5.0).get('scene_names', [])
        if available and target_scene not in available:
            raise ValueError(f'模拟器没有场景 {target_scene!r}，可选场景：{", ".join(available)}')
        sim.send('load_scene', scene_name=target_scene)
        pkt = sim.receive('telemetry', timeout=timeout)
        sim.control(0.0, 0.0, 1.0)
        for _ in range(3):
            pkt = sim.receive('telemetry', timeout=5.0)
        return sock, sim, pkt
    except Exception:
        try:
            sock.close()
        except OSError:
            pass
        raise


def run_closed_loop(
    host='127.0.0.1',
    port=9091,
    scene='generated_road',
    weights=None,
    model=None,
    throttle=0.10,
    max_steps=600,
    max_cte=0.0,
    stop_event=None,
    on_step=None,
    frame_every=1,
):
    """连接本机 DonkeySim，使用 RGB 模型执行真实闭环驾驶。

    参数:
        max_cte: 大于 0 时，若 |cte| 超过该阈值则提前结束；0 表示不设 CTE 强停。
        on_step: 每步回调函数，接收实时遥测与 JPEG base64 字典。
    """
    if model is None:
        model, weights_path = load_rgb_steering_model(weights)
    else:
        model.train_mode(False)
        weights_path = Path(weights) if weights else DEFAULT_WEIGHTS

    throttle = float(throttle)
    max_steps = max(1, int(max_steps))
    max_cte_lim = float(max_cte or 0.0)
    frame_every = max(1, int(frame_every))

    reason = 'steps_limit'
    step = 0
    max_abs_cte = 0.0
    travel = 0.0
    prev_pos = None
    last_hit = 'none'
    t0 = time.perf_counter()

    sock, sim, packet = _connect_and_enter_scene(host, port, scene)
    with sock:
        try:
            # 若模拟器默认出生点严重偏离赛道中心线（如 generated_road 初始 |cte| > 4），自动对齐到赛道中线起点
            if 'totalNodes' in packet and int(packet.get('totalNodes', 0)) >= 10:
                try:
                    node_count = int(packet['totalNodes'])
                    route = Route(sim.read_route(node_count))
                    init_pos = np.array([float(packet['pos_x']), float(packet['pos_z'])])
                    _, init_err = route.follow(init_pos, float(packet['yaw']), float(packet['speed']))
                    if init_err > 0.30:
                        spawn_pt = sim.route_node(int(packet.get('activeNode', 0)))
                        height = float(np.clip(float(packet['pos_y']) * WORLD_SCALE - spawn_pt[1], 0.33, 1.4))
                        packet = restart_at_node(sim, 5, height, 0.0, 0.0)
                except Exception:
                    pass
            for step_idx in range(1, max_steps + 1):
                if stop_event is not None and stop_event.is_set():
                    reason = 'stopped'
                    break

                pos_x = float(packet.get('pos_x', 0.0))
                pos_z = float(packet.get('pos_z', 0.0))
                yaw = float(packet.get('yaw', 0.0))
                speed = float(packet.get('speed', 0.0))
                cte = float(packet.get('cte', 0.0))
                hit = str(packet.get('hit', 'none'))
                last_hit = hit

                if not all(math.isfinite(v) for v in (pos_x, pos_z, yaw, speed, cte)):
                    raise ValueError(f'第 {step_idx} 步收到非有限遥测数值')

                if prev_pos is not None:
                    travel += math.hypot(pos_x - prev_pos[0], pos_z - prev_pos[1])
                prev_pos = (pos_x, pos_z)
                max_abs_cte = max(max_abs_cte, abs(cte))

                img, jpeg_b64 = _decode_rgb_and_jpeg(packet)
                t_inf = time.perf_counter()
                frame_tensor = prepare_rgb_frame(img)[None, ...]
                steering = float(model(constant(frame_tensor)).data.item())
                infer_ms = (time.perf_counter() - t_inf) * 1000.0
                if not math.isfinite(steering):
                    raise ValueError(f'第 {step_idx} 步模型输出非有限转向值')

                step = step_idx
                if hit != 'none':
                    reason = f'collision:{hit}'
                    sim.control(0.0, 0.0, 1.0)
                    if on_step is not None:
                        elapsed = time.perf_counter() - t0
                        on_step({
                            'state': 'running',
                            'scene': scene,
                            'step': step,
                            'max_steps': max_steps,
                            'steering': round(steering, 4),
                            'throttle': round(throttle, 3),
                            'speed': round(speed, 3),
                            'cte': round(cte, 4),
                            'max_abs_cte': round(max_abs_cte, 4),
                            'pos_x': round(pos_x, 3),
                            'pos_z': round(pos_z, 3),
                            'yaw': round(yaw, 2),
                            'hit': hit,
                            'travel': round(travel, 2),
                            'infer_ms': round(infer_ms, 2),
                            'fps': round(step / max(1e-6, elapsed), 1),
                            'elapsed_s': round(elapsed, 2),
                            'crop_top_fraction': CROP_TOP_FRACTION,
                            'image_b64': jpeg_b64,
                            'reason': reason,
                        })
                    break

                if max_cte_lim > 0 and abs(cte) > max_cte_lim:
                    reason = 'cte_exceeded'
                    sim.control(0.0, 0.0, 1.0)
                    break

                sim.control(steering, throttle, 0.0)
                elapsed = time.perf_counter() - t0

                if on_step is not None:
                    payload = {
                        'state': 'running',
                        'scene': scene,
                        'step': step,
                        'max_steps': max_steps,
                        'steering': round(steering, 4),
                        'throttle': round(throttle, 3),
                        'speed': round(speed, 3),
                        'cte': round(cte, 4),
                        'max_abs_cte': round(max_abs_cte, 4),
                        'pos_x': round(pos_x, 3),
                        'pos_z': round(pos_z, 3),
                        'yaw': round(yaw, 2),
                        'hit': hit,
                        'travel': round(travel, 2),
                        'infer_ms': round(infer_ms, 2),
                        'fps': round(step / max(1e-6, elapsed), 1),
                        'elapsed_s': round(elapsed, 2),
                        'crop_top_fraction': CROP_TOP_FRACTION,
                        'reason': '',
                    }
                    if step == 1 or step % frame_every == 0 or step == max_steps:
                        payload['image_b64'] = jpeg_b64
                    on_step(payload)

                if step_idx < max_steps:
                    packet = sim.receive('telemetry', timeout=5.0)
        finally:
            try:
                sim.control(0.0, 0.0, 1.0)
            except OSError:
                pass

    elapsed_s = time.perf_counter() - t0
    return {
        'scene': scene,
        'weights': str(weights_path),
        'input_shape': [3, RGB_HEIGHT, RGB_WIDTH],
        'steps': step,
        'max_steps': max_steps,
        'throttle': throttle,
        'max_abs_cte': round(max_abs_cte, 4),
        'travel': round(travel, 2),
        'hit': last_hit,
        'elapsed_s': round(elapsed_s, 2),
        'fps': round(step / max(1e-6, elapsed_s), 1),
        'reason': reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9091)
    parser.add_argument('--scene', default='generated_road', help='场景名，或填 current 使用当前赛道')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--throttle', type=float, default=0.10)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--max-cte', type=float, default=0.0, help='>0 时超过该 |CTE| 提前停止')
    parser.add_argument('--probe', action='store_true', help='仅探测本机 DonkeySim 与场景列表')
    args = parser.parse_args()

    if args.probe:
        print(json.dumps(probe_donkeysim(args.host, args.port), ensure_ascii=False, indent=2))
        return

    def _printer(st):
        if st['step'] == 1 or st['step'] % 25 == 0 or st.get('reason'):
            print(
                f"step={st['step']:04d}/{st['max_steps']} steer={st['steering']:+.3f} "
                f"cte={st['cte']:+.3f} speed={st['speed']:.2f} "
                f"infer={st['infer_ms']:.2f}ms fps={st['fps']:.1f} hit={st['hit']}",
                flush=True,
            )

    summary = run_closed_loop(
        host=args.host,
        port=args.port,
        scene=args.scene,
        weights=args.weights,
        throttle=args.throttle,
        max_steps=args.steps,
        max_cte=args.max_cte,
        on_step=_printer,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
