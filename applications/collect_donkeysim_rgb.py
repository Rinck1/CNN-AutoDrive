"""从本机 DonkeySim 采集彩色图像和路线跟踪控制标签。

打开模拟器并停留在赛道选择菜单。默认只采集一个回合。指定 --total-frames
可从同一场景的不同位置采集多个回合，并接着已有的完整回合继续。
每个回合保存 JPEG、samples.csv 和 summary.json。
"""

import argparse
import base64
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import socket
import time
from uuid import uuid4

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORLD_SCALE = 8.0  # 模拟器的路线节点是 Unity 坐标，相机遥测位置已除以 8。


class Simulator:
    """DonkeySim 的本机连接，保留相同遥测包里的 JPEG 与车辆状态。"""

    def __init__(self, sock, stream):
        self.sock = sock
        self.stream = stream

    def send(self, kind, **fields):
        # DonkeySim 接收单个 JSON 对象，返回以换行分隔的消息。
        self.sock.sendall(json.dumps({"msg_type": kind, **fields}).encode("ascii"))

    def receive(self, kind, timeout=10):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"等待模拟器消息 {kind} 超时")
            self.sock.settimeout(remaining)
            try:
                line = self.stream.readline()
            except TimeoutError as exc:
                raise TimeoutError(f"等待模拟器消息 {kind} 超时") from exc
            if not line:
                raise ConnectionError("模拟器已断开连接")
            message = json.loads(line)
            if message.get("msg_type") == kind:
                return message

    def control(self, steering, throttle, brake=0.0):
        self.send("control", steering=str(steering), throttle=str(throttle), brake=str(brake))

    def scenes(self):
        self.receive("scene_selection_ready")
        self.send("get_scene_names")
        return self.receive("scene_names")["scene_names"]

    def read_route(self, node_count):
        # 每隔一个节点取样，保留末端点。逐条请求可避免 TCP 消息合并。
        indices = list(range(0, node_count - 1, 2)) + [node_count - 1]
        return np.asarray([self.route_node(index)[[0, 2]] for index in indices]) / WORLD_SCALE

    def route_node(self, index):
        self.send("node_position", index=str(index))
        node = self.receive("node_position", timeout=3)
        if int(node["index"]) != index:
            raise ValueError(f"模拟器返回的路线节点编号与请求不符：{node['index']}")
        return np.array([float(node[key]) for key in ("pos_x", "pos_y", "pos_z")])


class Route:
    """找出车辆在路线上的投影点，再跟踪它前方的一个目标点。"""

    def __init__(self, points):
        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        points = points[np.r_[True, gaps > 1e-6]]
        if len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("赛道的路线节点不足或无效")
        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        closing_gap = np.linalg.norm(points[-1] - points[0])
        self.loop = len(points) >= 8 and closing_gap <= 2.5 * np.median(gaps)
        if self.loop and closing_gap > 1e-6:
            points = np.concatenate((points, points[:1]))
        self.points = points
        self.deltas = np.diff(points, axis=0)
        self.lengths = np.linalg.norm(self.deltas, axis=1)
        self.distances = np.r_[0.0, np.cumsum(self.lengths)]

    def follow(self, pos, yaw, speed):
        fractions = np.clip(
            np.sum((pos - self.points[:-1]) * self.deltas, axis=1) / self.lengths**2,
            0.0,
            1.0,
        )
        projections = self.points[:-1] + fractions[:, None] * self.deltas
        deviations = np.linalg.norm(pos - projections, axis=1)
        nearest = int(np.argmin(deviations))
        distance = self.distances[nearest] + fractions[nearest] * self.lengths[nearest]
        target_distance = distance + float(np.clip(0.60 + 0.30 * speed, 0.60, 1.5))
        if self.loop:
            target_distance %= self.distances[-1]
        elif distance >= self.distances[-1] - 0.30:
            return None, float(deviations[nearest])
        target = np.array([
            np.interp(target_distance, self.distances, self.points[:, axis]) for axis in (0, 1)
        ])
        dx, dz = target - pos
        bearing = math.atan2(dx, dz) - math.radians(yaw)
        # Pure pursuit：用前视目标的相对方向计算曲率，再换算成模拟器转向量。
        curvature = 2.0 * math.sin(bearing) / max(math.hypot(dx, dz), 0.15)
        steer = math.atan(0.28 * curvature) / math.radians(16.0)
        return float(np.clip(steer, -1.0, 1.0)), float(deviations[nearest])


def frame_digest(raw):
    return hashlib.sha256(raw).digest()


def save_frame(path, packet, seen=None):
    raw = base64.b64decode(packet["image"], validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        size = image.size
        if image.format != "JPEG" or image.mode != "RGB":
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=95)
            raw = buffer.getvalue()
    if seen is not None:
        digest = frame_digest(raw)
        if digest in seen:
            return None
    path.write_bytes(raw)
    if seen is not None:
        seen.add(digest)
    return size


def collect(sim, args, route, initial, final_dir, limit=None, seen=None, spawn=None):
    if final_dir.exists():
        raise FileExistsError(f"采集目录已存在：{final_dir}")
    out = final_dir.with_name(f"{final_dir.name}.partial-{uuid4().hex[:8]}")
    out.mkdir(parents=True, exist_ok=False)
    images = out / "images"
    images.mkdir()
    episode = f"{args.scene}-{uuid4().hex[:12]}"
    fields = ("image", "scene", "episode", "frame", "sim_time", "steering", "throttle",
              "speed", "cte", "label_source")
    count = 0
    steps = 0
    image_size = None
    max_cte = 0.0
    positions = []
    reason = "steps_limit"
    packet = initial
    last_saved_time = -math.inf

    with (out / "samples.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for frame in range(args.steps):
            pos = np.array([float(packet["pos_x"]), float(packet["pos_z"])])
            yaw, speed, cte, stamp = (float(packet[key]) for key in ("yaw", "speed", "cte", "time"))
            if not all(math.isfinite(v) for v in (*pos, yaw, speed, cte, stamp)):
                raise ValueError(f"第 {frame} 帧的车辆遥测包含非有限数值")
            positions.append(pos)
            max_cte = max(max_cte, abs(cte))
            steering, error = route.follow(pos, yaw, speed)
            if packet["hit"] != "none":
                reason = "collision"
                break
            if error > 0.35:
                reason = "away_from_route"
                break
            if steering is None:
                reason = "end_of_route"
                break

            sim.control(steering, args.throttle)
            steps += 1
            if frame % args.every == 0 and speed >= 0.08 and stamp > last_saved_time:
                # 保存作出本次转向前的图像，配对刚刚发出的控制量。
                image_path = Path("images") / f"{frame:06d}_{steering:+.6f}.jpg"
                size = save_frame(out / image_path, packet, seen)
                if size is not None:
                    image_size = size
                    writer.writerow(dict(image=image_path.as_posix(), scene=args.scene, episode=episode,
                                         frame=frame, sim_time=stamp, steering=steering,
                                         throttle=args.throttle, speed=speed, cte=cte,
                                         label_source="sim_centerline_controller"))
                    count += 1
                    last_saved_time = stamp
                    if count == limit:
                        reason = "frames_limit"
                        break
            if frame + 1 < args.steps:
                packet = sim.receive("telemetry", timeout=5)

    route_travel = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    summary = dict(scene=args.scene, episode=episode, frames=count, steps=steps, reason=reason,
                   throttle=args.throttle, max_abs_cte=max_cte, travel=route_travel,
                   resolution=image_size, every=args.every, label_source="sim_centerline_controller")
    if spawn is not None:
        summary.update(spawn)
    if limit is not None and count == 0:
        (out / "samples.csv").unlink()
        images.rmdir()
        out.rmdir()
        print(json.dumps(summary), flush=True)
        return summary
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    out.rename(final_dir)
    print(json.dumps(summary), flush=True)
    print(f"数据目录：{final_dir.resolve()}", flush=True)
    return summary


def restart_at_node(sim, index, height, offset, yaw_offset):
    """在静止状态把车放到指定路段，等新位置的相机画面稳定后开始驾驶。"""
    sim.control(0, 0, 1)
    sim.send("reset_car")
    for _ in range(6):
        sim.receive("telemetry", timeout=5)
    point = sim.route_node(index)
    ahead = sim.route_node(index + 1)
    dx, dz = (ahead - point)[[0, 2]]
    heading = math.atan2(dx, dz)
    x = point[0] + offset * WORLD_SCALE * math.cos(heading)
    z = point[2] - offset * WORLD_SCALE * math.sin(heading)
    yaw = heading + math.radians(yaw_offset)
    sim.send("set_position", pos_x=str(x), pos_y=str(point[1] + height), pos_z=str(z),
             Qx="0", Qy=str(math.sin(yaw / 2)), Qz="0", Qw=str(math.cos(yaw / 2)))
    near = 0
    for _ in range(50):
        packet = sim.receive("telemetry", timeout=5)
        error = math.hypot(float(packet["pos_x"]) * WORLD_SCALE - x,
                           float(packet["pos_z"]) * WORLD_SCALE - z)
        near = near + 1 if error <= 1.0 else 0
        if near >= 8 and packet["hit"] == "none":
            return packet
    raise ValueError(f"无法在路线节点 {index} 上获得稳定相机画面")


def existing_episodes(root, scene):
    """恢复上次已经完整写入的帧，同时核对回合归属与文件数量。"""
    kept = []
    seen = set()
    for directory in sorted(root.glob("episode_*")):
        if not directory.is_dir() or not directory.name.removeprefix("episode_").isdigit():
            continue
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["scene"] != scene:
            raise ValueError(f"回合的场景与当前场景不一致：{directory}")
        with (directory / "samples.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != summary["frames"]:
            raise ValueError(f"回合的图片记录数量不一致：{directory}")
        for row in rows:
            image = directory / row["image"]
            digest = frame_digest(image.read_bytes())
            if digest in seen:
                raise ValueError(f"已有回合存在完全重复的图片：{image}")
            seen.add(digest)
        kept.append(int(directory.name.removeprefix("episode_")))
    return max(kept, default=0), seen


def collect_batch(sim, args, route, initial, node_count, episode_number, seen):
    total = args.total_frames
    spawn_point = sim.route_node(int(initial["activeNode"]))
    height = float(np.clip(float(initial["pos_y"]) * WORLD_SCALE - spawn_point[1], 0.33, 1.4))
    # 交错选取赛道前、中、后部的起点，并少量改变横向位置和车头方向。
    sections = np.linspace(0.06, 0.90, 8)
    short_runs = 0
    while len(seen) < total:
        episode_number += 1
        rng = np.random.default_rng(args.seed + episode_number)
        section = sections[(episode_number - 2) % len(sections)]
        node = int(np.clip((section + rng.uniform(-0.045, 0.045)) * node_count,
                           2, node_count - 3))
        offset = float(rng.uniform(-0.065, 0.065))
        yaw_offset = float(rng.uniform(-5, 5))
        remaining = total - len(seen)
        spawn = dict(start_node=node, start_offset=offset, start_yaw_offset_degrees=yaw_offset,
                     seed=args.seed)
        path = args.output_dir / f"episode_{episode_number:04d}"
        try:
            frame = restart_at_node(sim, node, height, offset, yaw_offset)
        except ValueError as exc:
            print(str(exc), flush=True)
            frames = 0
        else:
            info = collect(sim, args, route, frame, path, limit=remaining, seen=seen, spawn=spawn)
            frames = info["frames"]
        print(f"{args.scene}：{len(seen)}/{total}，本回合 {frames} 张", flush=True)
        short_runs = 0 if frames >= 10 else short_runs + 1
        if short_runs >= 8 and len(seen) < total:
            raise ValueError("连续 8 次未采到足够画面，请检查可用起点和赛道状态")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scene", help="模拟器场景名称，如 generated_road")
    mode.add_argument("--list-scenes", action="store_true", help="列出本机模拟器的可选场景")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--steps", type=int, default=300, help="最多发送多少次驾驶控制")
    parser.add_argument("--every", type=int, default=3, help="每隔多少个控制步保存一帧")
    parser.add_argument("--throttle", type=float, default=0.1, help="本回合的固定油门")
    parser.add_argument("--output-dir", type=Path, help="单回合的目录，或多回合采集的场景目录")
    parser.add_argument("--total-frames", type=int, help="按多个回合采集的总张数")
    parser.add_argument("--seed", type=int, default=42, help="多回合起点扰动的随机种子")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.steps < 1 or args.every < 1 or not 0 < args.throttle <= 1:
        parser.error("端口须在 1～65535，steps 和 every 须为正整数，油门须在 (0, 1] 内")
    if args.total_frames is not None and (args.total_frames < 1 or args.seed < 0 or args.output_dir is None):
        parser.error("多回合采集需要提供输出目录、正整数目标张数和非负随机种子")
    try:
        if args.total_frames is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            episode_number, seen = existing_episodes(args.output_dir, args.scene)
            if len(seen) > args.total_frames:
                parser.error(f"已有 {len(seen)} 张，超过目标 {args.total_frames} 张")
            print(f"{args.scene}：已有 {len(seen)}/{args.total_frames} 张", flush=True)
            if len(seen) == args.total_frames:
                return
        with socket.create_connection(("127.0.0.1", args.port), timeout=5) as sock, sock.makefile("rb") as stream:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sim = Simulator(sock, stream)
            scenes = sim.scenes()
            if args.list_scenes:
                print("\n".join(scenes))
                return
            if args.scene not in scenes:
                parser.error(f"模拟器没有场景 {args.scene!r}。可选场景：{', '.join(scenes)}")
            if args.output_dir is None:
                args.output_dir = (ROOT / "data" / "donkeysim_rgb" / args.scene
                                   / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            if args.total_frames is None and args.output_dir.exists():
                parser.error(f"采集目录已存在：{args.output_dir}")
            print(f"加载场景：{args.scene}", flush=True)
            sim.send("load_scene", scene_name=args.scene)
            try:
                initial = sim.receive("telemetry", timeout=20)
                sim.control(0, 0, 1)
                if not all(key in initial for key in ("totalNodes", "pos_x", "pos_z", "cte", "time")):
                    raise ValueError("场景缺少路线或位置数据，无法生成可靠的转向标签")
                node_count = int(initial["totalNodes"])
                if node_count < 3:
                    raise ValueError("此场景没有足够的路线节点")
                route = Route(sim.read_route(node_count))
                print(f"已读取路线（{node_count} 个原始节点）", flush=True)
                frame = sim.receive("telemetry")
                if args.total_frames is None:
                    info = collect(sim, args, route, frame, args.output_dir)
                    if info["frames"] == 0:
                        raise ValueError("本回合没有采到车辆移动时的有效画面")
                elif node_count >= 7:
                    collect_batch(sim, args, route, frame, node_count, episode_number, seen)
                else:
                    raise ValueError("路线节点不足，无法选择足够的独立起点")
            finally:
                try:
                    sim.control(0, 0, 1)
                except OSError:
                    pass
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"采集失败：{exc}\n")


if __name__ == "__main__":
    main()
