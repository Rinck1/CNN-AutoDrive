"""用 MiniFlows 的转向模型控制课堂版 DonkeySim，并可保存相机演示。"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.donkeycar import IMG_SIZE, build_model
from miniflows.graph import constant
from miniflows.utils import load_model


def prepare_frame(rgb):
    """将 DonkeySim 的 RGB 相机帧转换为现有 CNN 需要的灰度张量。"""
    gray = Image.fromarray(rgb).convert("L")
    resized = gray.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None, None] / 255.0


def replay_frame(rgb, step, steering, throttle):
    """在相机回放上标出这一帧对应的模型动作。"""
    frame = Image.fromarray(rgb).resize((480, 360), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 480, 28), fill="black")
    draw.text((8, 8), f"MiniFlows | step {step} | steer {steering:+.3f} | throttle {throttle:.2f}",
              fill="white")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "outputs/donkeycar/donkeycar_model.npz")
    parser.add_argument("--track", default="donkey-generated-roads-v0")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--throttle", type=float, default=0.1)
    parser.add_argument("--record", type=Path, help="保存相机回放到 GIF")
    args = parser.parse_args()
    if args.steps <= 0 or not 0 < args.throttle <= 1:
        parser.error("steps 必须为正整数，throttle 必须在 (0, 1] 内")
    if args.record is not None and args.record.suffix.lower() != ".gif":
        parser.error("--record 需要 .gif 路径")
    if not args.weights.is_file():
        parser.error(f"找不到权重 {args.weights}，请先运行 applications/donkeycar.py")

    import gymnasium as gym
    import gym_donkeycar  # 注册课堂模拟器环境

    model = load_model(build_model(), args.weights)
    model.train_mode(False)
    print(f"Connecting to {args.track} at 127.0.0.1:{args.port}")
    env = gym.make(args.track, conf={"host": "127.0.0.1", "port": args.port})
    frames = []
    errors = []
    reason = "step limit"
    try:
        rgb, _ = env.reset()
        print(f"Camera: {rgb.shape}, {rgb.dtype}")
        for step in range(1, args.steps + 1):
            steering = model(constant(prepare_frame(rgb))).data.item()
            if not math.isfinite(steering):
                raise ValueError(f"模型在第 {step} 步返回非有限转向值")
            action = np.array([steering, args.throttle], dtype=np.float32)
            rgb, _, terminated, truncated, info = env.step(action)
            if "cte" in info:
                errors.append(abs(float(info["cte"])))
            if args.record is not None:
                frames.append(replay_frame(rgb, step, steering, args.throttle))
            if step == 1 or step % 25 == 0 or terminated or truncated:
                print(f"step={step} steer={steering:+.3f} cte={info.get('cte')} hit={info.get('hit')}")
            if terminated or truncated:
                reason = "simulator terminated" if terminated else "simulator truncated"
                break
    finally:
        env.close()

    max_error = f"{max(errors):.3f}" if errors else "unavailable"
    print(f"Run ended: {reason}; model steps={step}; max |cte|={max_error}")
    if frames:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.record, save_all=True, append_images=frames[1:],
                       duration=50, loop=0)
        print(f"Replay: {args.record.resolve()}")


if __name__ == "__main__":
    main()
