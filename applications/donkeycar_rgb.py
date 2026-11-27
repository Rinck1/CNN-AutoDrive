"""用 DonkeySim 彩色相机图像训练 MiniFlows 转向回归模型。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniflows.graph import constant
from miniflows.layers import Conv2D, Flatten, Linear, ReLU, Sequential, Tanh
from miniflows.losses import MSELoss
from miniflows.optim import Adam
from miniflows.trainer import Trainer
from miniflows.utils import count_params, load_model, save_model, set_seed


MANIFEST = ROOT / 'data' / 'donkeysim_rgb' / 'three_scenes_10000' / 'samples.csv'
OUTPUTS = ROOT / 'outputs' / 'donkeycar_rgb'
WIDTH, HEIGHT = 64, 40
CROP_TOP_FRACTION = 1 / 6


@dataclass(frozen=True)
class Sample:
    image: str
    episode: str
    steering: float


def read_samples(manifest):
    """读取相机图像路径、分组编号和转向标签。"""
    samples = []
    with manifest.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if not {'image', 'episode', 'steering'}.issubset(reader.fieldnames or ()):
            raise ValueError('样本清单需要 image、episode、steering 三列')
        for row in reader:
            image = (row['image'] or '').strip()
            episode = (row['episode'] or '').strip()
            try:
                value = float(row['steering'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'样本清单第 {reader.line_num} 行的转向值无法解析') from exc
            if not image or not episode or not -1 <= value <= 1:
                raise ValueError(f'样本清单第 {reader.line_num} 行图像、回合或转向值无效')
            samples.append(Sample(image, episode, value))
    if len({sample.image for sample in samples}) != len(samples):
        raise ValueError('样本清单中有重复图像路径')
    return samples


def split_episodes(samples, seed):
    """随机分配完整驾驶回合，约 70% 训练、15% 验证、15% 测试。"""
    episodes = list(dict.fromkeys(sample.episode for sample in samples))
    if len(episodes) < 3:
        raise ValueError('至少需要三个独立记录的驾驶回合')
    order = np.random.default_rng(seed).permutation(len(episodes))
    size = max(1, round(len(episodes) * 0.15))
    test = {episodes[i] for i in order[:size]}
    validation = {episodes[i] for i in order[size:2 * size]}
    partitions = {'train': [], 'validation': [], 'test': []}
    for sample in samples:
        if sample.episode in test:
            partitions['test'].append(sample)
        elif sample.episode in validation:
            partitions['validation'].append(sample)
        else:
            partitions['train'].append(sample)
    return partitions


def prepare_frame(image):
    """保留 RGB，裁掉顶部约 1/6 后缩小为 (3, 40, 64)。"""
    width, height = image.size
    top = int(height * CROP_TOP_FRACTION)
    road_view = image.crop((0, top, width, height)).convert('RGB')
    resized = road_view.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    return np.moveaxis(pixels, -1, 0) / 255.0 - 0.5


def read_frame(path):
    with Image.open(path) as image:
        return prepare_frame(image)


def load_partition(samples, data_root, workers, name):
    images = np.empty((len(samples), 3, HEIGHT, WIDTH), dtype=np.float32)
    labels = np.array([sample.steering for sample in samples], dtype=np.float64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # 分段提交，限制未消费的解码结果数量。
        for start in range(0, len(samples), 512):
            paths = (data_root / sample.image for sample in samples[start:start + 512])
            for index, image in enumerate(pool.map(read_frame, paths), start=start):
                images[index] = image
            done = min(len(samples), start + 512)
            if done % 4096 == 0 or done == len(samples):
                print(f'{name} 图像已读取 {done}/{len(samples)}', flush=True)
    return images, labels


def build_model():
    return Sequential(
        Conv2D(3, 8, 3, stride=2, padding=1, name='rgb_conv1'), ReLU(),
        Conv2D(8, 12, 3, stride=2, padding=1, name='rgb_conv2'), ReLU(),
        Conv2D(12, 16, 3, stride=2, padding=1, name='rgb_conv3'), ReLU(),
        Flatten(), Linear(16 * 5 * 8, 64, name='features'), ReLU(),
        Linear(64, 1, init='xavier', name='steering'), Tanh(),
    )


def regression_metrics(prediction, truth, train_mean):
    errors = prediction - truth
    absolute = np.abs(errors)
    turns = np.abs(truth) >= 0.10
    return {
        'mae': float(absolute.mean()),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'p95_absolute_error': float(np.quantile(absolute, 0.95)),
        'zero_steering_baseline_mae': float(np.abs(truth).mean()),
        'train_mean_baseline_mae': float(np.abs(truth - train_mean).mean()),
        'turn_direction_accuracy_for_abs_label_ge_0.10': (
            float(np.mean(np.sign(prediction[turns]) == np.sign(truth[turns])))
            if turns.any() else None
        ),
    }


def write_split(path, partitions):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['split', 'image', 'episode', 'steering'])
        for name, rows in partitions.items():
            for row in rows:
                writer.writerow([name, row.image, row.episode, row.steering])


def plot_report(path, history, best_epoch, target, prediction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (loss_ax, steer_ax) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    loss_ax.plot(epochs, history['train_loss'], label='Training')
    loss_ax.plot(epochs, history['val_loss'], label='Validation')
    loss_ax.axvline(best_epoch, color='#7992a9', linestyle=':', label='Selected epoch')
    loss_ax.set(xlabel='Epoch', ylabel='MSE', ylim=(0, None))
    loss_ax.legend()
    steer_ax.scatter(target, prediction, s=6, alpha=0.18, color='#165a91', edgecolors='none')
    bounds = [min(target.min(), prediction.min()), max(target.max(), prediction.max())]
    steer_ax.plot(bounds, bounds, color='#888888', linestyle='--', linewidth=1)
    steer_ax.set(xlabel='Recorded steering', ylabel='Predicted steering', title='Held-out episodes')
    steer_ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(args):
    samples = read_samples(args.manifest)
    partitions = split_episodes(samples, args.seed)
    extra_train_samples = []
    extra_roots = {}
    for extra_path in (args.extra_manifest or []):
        extra_rows = read_samples(extra_path)
        for rep in range(args.extra_repeat):
            for row in extra_rows:
                # 记录原始根目录，方便跨目录加载图像
                extra_train_samples.append((extra_path.parent, row))
        extra_roots[str(extra_path)] = len(extra_rows)
    out = args.output_dir or OUTPUTS / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    weights = out / 'steering_model.npz'
    split_counts = {name: {'images': len(rows), 'episodes': len({row.episode for row in rows})}
                    for name, rows in partitions.items()}
    if extra_train_samples:
        split_counts['extra_train'] = {'images': len(extra_train_samples), 'sources': extra_roots,
                                       'repeat': args.extra_repeat}
    write_split(out / 'split.csv', partitions)
    print(json.dumps(split_counts, ensure_ascii=False), flush=True)
    data_root = args.manifest.parent
    x_train, y_train = load_partition(partitions['train'], data_root, args.workers, 'train')
    if extra_train_samples:
        # 逐个额外清单只解码一次，再按 extra_repeat 拼接，节省 I/O
        x_extras, y_extras = [], []
        for extra_path in (args.extra_manifest or []):
            extra_rows = read_samples(extra_path)
            xe, ye = load_partition(extra_rows, extra_path.parent, args.workers, f'extra:{extra_path.parent.name}')
            for _ in range(args.extra_repeat):
                x_extras.append(xe)
                y_extras.append(ye)
        x_train = np.concatenate([x_train, *x_extras], axis=0)
        y_train = np.concatenate([y_train, *y_extras], axis=0)
    if args.flip_aug:
        # 水平镜像翻转增强（图片左右翻转 + 转向取反），消除系统性左右偏置并翻倍训练数据
        x_train = np.concatenate([x_train, x_train[:, :, :, ::-1].copy()], axis=0)
        y_train = np.concatenate([y_train, -y_train], axis=0)
        print(f'已启用水平镜像翻转增强：训练集扩充至 {len(x_train)} 张，转向均值={y_train.mean():+.5f}', flush=True)
    x_val, y_val = load_partition(partitions['validation'], data_root, args.workers, 'validation')
    train_mean = float(y_train.mean())
    set_seed(args.seed)
    model = build_model()
    optimizer = Adam(model.params(), lr=args.lr)
    trainer = Trainer(model, MSELoss(), optimizer, task='regression', seed=args.seed)
    start = time.perf_counter()
    epoch_end = start
    best_loss = math.inf
    best_epoch = 0

    def checkpoint(state):
        nonlocal best_loss, best_epoch, epoch_end
        epoch = state['epoch']
        val_loss = state['val_loss']
        if not math.isfinite(val_loss) or not math.isfinite(state['train_loss']):
            raise ValueError(f'第 {epoch} 轮出现非有限损失')
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            staging = out / 'steering_model_pending.npz'
            save_model(model, staging)
            staging.replace(weights)
        now = time.perf_counter()
        print(f'epoch {epoch:02d}/{args.epochs:02d}  train MSE={state["train_loss"]:.5f}  '
              f'val MSE={val_loss:.5f}  lr={optimizer.lr:g}  {now - epoch_end:.1f}s', flush=True)
        epoch_end = now
        # 后半程减少步长，便于把收敛后的转向误差降下来。
        if epoch == math.ceil(args.epochs * 0.65):
            optimizer.lr *= 0.4
        return False

    history = trainer.train(x_train, y_train, x_val, y_val, epochs=args.epochs,
                            batch_size=args.batch_size, callbacks=[checkpoint], verbose=False)
    training_seconds = time.perf_counter() - start
    del x_train, y_train
    load_model(model, weights)
    val_prediction = trainer.predict(x_val, batch_size=args.batch_size).ravel()
    x_test, y_test = load_partition(partitions['test'], data_root, args.workers, 'test')
    test_prediction = trainer.predict(x_test, batch_size=args.batch_size).ravel()
    if not np.isfinite(test_prediction).all() or not np.isfinite(val_prediction).all():
        raise ValueError('最优权重的验证或测试预测包含非有限值')
    report = {
        'manifest': str(args.manifest.resolve()),
        'model': str(weights.name),
        'framework': 'MiniFlows',
        'device': 'cpu',
        'input': {'channels': 'RGB', 'shape_chw': [3, HEIGHT, WIDTH],
                  'crop_top_fraction': CROP_TOP_FRACTION, 'pixel_transform': 'pixel / 255 - 0.5'},
        'target': 'steering',
        'steering_mean': {'dataset': float(np.mean([row.steering for row in samples])),
                          'train': train_mean},
        'parameters': count_params(model),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'best_epoch_by_validation_mse': best_epoch,
        'training_seconds': round(training_seconds, 2),
        'splits': split_counts,
        'validation': regression_metrics(val_prediction, y_val, train_mean),
        'test': regression_metrics(test_prediction, y_test, train_mean),
        'history': {'training_mse': history['train_loss'], 'validation_mse': history['val_loss'],
                    'learning_rate': history['lr']},
    }
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (out / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'episode', 'steering', 'predicted_steering'])
        for sample, predicted in zip(partitions['test'], test_prediction):
            writer.writerow([sample.image, sample.episode, sample.steering, float(predicted)])
    plot_report(out / 'training.png', history, best_epoch, y_test, test_prediction)
    print(json.dumps(report['test']), flush=True)
    print(f'已保存权重和训练结果：{out.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='action', required=True)
    train_parser = subparsers.add_parser('train', help='按驾驶回合切分并训练')
    train_parser.add_argument('--manifest', type=Path, default=MANIFEST)
    train_parser.add_argument('--extra-manifest', type=Path, action='append', default=[],
                              help='附加到训练集的纠偏/DAgger 样本清单（可多次传入）')
    train_parser.add_argument('--extra-repeat', type=int, default=1,
                              help='附加纠偏样本在训练集中的重复倍数')
    train_parser.add_argument('--flip-aug', action='store_true',
                              help='对训练集启用水平镜像翻转增强（转向取反）')
    train_parser.add_argument('--output-dir', type=Path)
    train_parser.add_argument('--epochs', type=int, default=10)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--workers', type=int, default=6)
    train_parser.add_argument('--lr', type=float, default=0.001)
    predict = subparsers.add_parser('predict', help='从一张 RGB 图片预测转向')
    predict.add_argument('image', type=Path)
    predict.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'train':
            if min(args.epochs, args.batch_size, args.workers) < 1 or not 0 <= args.seed < 2**32:
                parser.error('轮数、批大小、解码线程数需为正数，随机种子应为 0 到 2^32−1')
            if not math.isfinite(args.lr) or args.lr <= 0:
                parser.error('学习率必须为有限正数')
            train(args)
        else:
            model = build_model()
            with np.load(args.weights, allow_pickle=False) as archive:
                if set(archive.files) != {name for name, _ in model.named_params()}:
                    raise ValueError('权重文件与 RGB 转向模型不匹配')
            load_model(model, args.weights)
            model.train_mode(False)
            output = model(constant(read_frame(args.image)[None, ...])).data.item()
            print(json.dumps({'steering': output}))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
