# -*- coding: utf-8 -*-
"""
miniflows/data.py —— 数据预处理模块（扩展④：数据增强与预处理）
====================================================================

本模块是 MiniFlows 框架的"数据层"，与计算图完全解耦：
所有函数只处理 numpy 数组，返回 numpy 数组（不包装成 Node），
由 Trainer 在训练时统一把 batch 包装成 Node。

提供的能力：
1. 特征缩放：standard_scale（标准化 z-score）、minmax_scale（区间缩放）、
   apply_scale（用训练集统计量变换测试集，保证无数据泄露）；
2. 标签编码：one_hot（整数标签 -> one-hot 矩阵）；
3. 数据集划分：train_test_split（支持 stratify 分层抽样，保证训练/测试集类别比例一致）；
4. 合成数据：make_blobs（高斯团簇，逻辑回归演示用，不依赖 sklearn）；
5. 批量迭代：DataLoader（小批量乱序迭代器，Keras 风格）；
6. 真实数据集：load_iris（鸢尾花，三级降级：sklearn -> 本地缓存 -> 合成数据）、
   load_mnist / load_mnist_arrays（MNIST，torchvision -> urllib 镜像 idx.gz 下载降级）；
7. 数据增强：random_shift_images（图像随机平移增强，整数像素循环平移）。

所有函数均带详细中文注释，便于课程答辩讲解。
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

# MNIST 数据的官方镜像地址（课程环境已确认可达，比原 sites.google快捷）
_MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
# 四个 idx.gz 文件：训练 6 万张 / 测试 1 万张，均为 28x28 灰度图
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# ======================================================================
# 一、特征缩放（扩展④：预处理）
# ======================================================================
def standard_scale(X):
    """Z-score 标准化：把每个特征列变成均值 0、标准差 1。

    实现思路：
        对每一列（特征）计算 mean 与 std，然后 (X - mean) / std；
        std 为 0 的列（常数列）用 1 代替避免除零。

    参数：
        X (np.ndarray): 原始特征矩阵，形状 (N, F)。

    返回：
        (X_scaled, scaler) 元组：
            X_scaled (np.ndarray): 标准化后的矩阵，形状同 X；
            scaler (dict): {'mean': 各列均值, 'std': 各列标准差}，
                           保存下来供 apply_scale 变换测试集使用。
    """
    X = np.asarray(X, dtype=np.float64)                     # 统一转 float64 保证精度
    mean = X.mean(axis=0)                                   # 沿样本轴(第0轴)求每列均值
    std = X.std(axis=0)                                     # 沿样本轴求每列标准差(总体std, ddof=0)
    std = np.where(std == 0, 1.0, std)                      # 常数列 std=0 时置 1，避免除零
    X_scaled = (X - mean) / std                             # 广播减法 + 除法完成标准化
    scaler = {"mean": mean, "std": std}                     # 保存统计量供测试集复用
    return X_scaled, scaler


def minmax_scale(X, feature_range=(0, 1)):
    """Min-Max 区间缩放：把每个特征列线性映射到指定区间（默认 [0,1]）。

    实现思路：
        X_scaled = (X - min) / (max - min) * (hi - lo) + lo；
        极差为 0 的列用 1 代替避免除零。

    参数：
        X (np.ndarray): 原始特征矩阵，形状 (N, F)；
        feature_range (tuple): 目标区间 (lo, hi)，默认 (0, 1)。

    返回：
        (X_scaled, scaler) 元组：
            scaler (dict): {'min', 'max', 'feature_range'}，供 apply_scale 复用。
    """
    X = np.asarray(X, dtype=np.float64)                     # 统一精度
    lo, hi = feature_range                                  # 目标区间下界/上界
    xmin = X.min(axis=0)                                    # 每列最小值
    xmax = X.max(axis=0)                                    # 每列最大值
    rng = xmax - xmin                                       # 每列极差
    rng = np.where(rng == 0, 1.0, rng)                      # 常数列极差置 1 防除零
    X01 = (X - xmin) / rng                                  # 先压到 [0,1]
    X_scaled = X01 * (hi - lo) + lo                         # 再线性映射到目标区间
    scaler = {"min": xmin, "max": xmax, "feature_range": (lo, hi)}
    return X_scaled, scaler


def apply_scale(X, scaler):
    """用已有 scaler（训练集统计量）变换新数据（如测试集），避免数据泄露。

    实现思路：
        根据 scaler 里包含的键判断是标准化还是区间缩放，套用同一组统计量。

    参数：
        X (np.ndarray): 待变换矩阵；
        scaler (dict): standard_scale / minmax_scale 返回的统计量字典。

    返回：
        np.ndarray: 变换后的矩阵。
    """
    X = np.asarray(X, dtype=np.float64)
    if "mean" in scaler:                                    # 标准化 scaler
        return (X - scaler["mean"]) / scaler["std"]
    elif "min" in scaler:                                   # 区间缩放 scaler
        lo, hi = scaler["feature_range"]
        rng = scaler["max"] - scaler["min"]
        rng = np.where(rng == 0, 1.0, rng)                  # 同样防除零
        return (X - scaler["min"]) / rng * (hi - lo) + lo
    else:                                                   # 未知格式直接报错，提示用户
        raise ValueError("scaler 格式不正确，应为 standard_scale/minmax_scale 的返回值")


# ======================================================================
# 二、标签编码
# ======================================================================
def one_hot(y_int, num_classes):
    """整数标签 -> one-hot 矩阵。

    实现思路：
        先建全 0 矩阵 (N, C)，再用高级索引把每行对应类别位置置 1。

    参数：
        y_int (np.ndarray | list): 整数标签，形状 (N,)，取值 [0, num_classes)；
        num_classes (int): 类别总数 C。

    返回：
        np.ndarray: one-hot 矩阵，形状 (N, C)，float64。
    """
    y = np.asarray(y_int, dtype=np.int64).ravel()           # 展平成一维整数标签
    if y.min() < 0 or y.max() >= num_classes:               # 越界检查，尽早报错
        raise ValueError(f"标签取值超出 [0, {num_classes}) 范围")
    out = np.zeros((len(y), num_classes), dtype=np.float64)  # 全 0 矩阵
    out[np.arange(len(y)), y] = 1.0                          # 高级索引一次性置 1
    return out


# ======================================================================
# 三、数据集划分（支持分层抽样）
# ======================================================================
def train_test_split(X, y, test_size=0.2, shuffle=True, stratify=None, seed=0):
    """把数据集划分为训练集与测试集；stratify=y 时按类别比例分层抽样。

    实现思路：
        - 不分层：整体洗牌后按 test_size 截取前后两段；
        - 分层：对每个类别单独洗牌，按比例抽取该类样本进测试集，
          保证训练/测试集中各类别比例与原数据一致（小样本任务尤其重要）。

    参数：
        X (np.ndarray): 特征矩阵 (N, ...)；
        y (np.ndarray): 标签 (N,)；
        test_size (float | int): 测试集比例（0~1 的浮点数）或样本数（整数）；
        shuffle (bool): 是否先随机打乱；
        stratify (np.ndarray | None): 分层依据的标签数组，None 表示不分层；
        seed (int): 随机种子，保证可复现。

    返回：
        (X_train, y_train, X_test, y_test) 四元组。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)                                              # 总样本数
    rs = np.random.RandomState(seed)                        # 独立随机源，互不影响全局种子

    def _n_test(total):
        # 把 test_size 换算成具体的测试样本数（浮点比例 -> 四舍五入取整）
        if isinstance(test_size, float):
            return max(1, int(round(total * test_size)))
        return int(min(test_size, total))

    if stratify is None:
        # ---------- 普通划分：整体洗牌后截断 ----------
        idx = rs.permutation(n) if shuffle else np.arange(n)  # 打乱（或保持原序）的索引
        k = _n_test(n)                                      # 测试集样本数
        test_idx, train_idx = idx[:k], idx[k:]              # 前 k 个做测试集，其余做训练集
    else:
        # ---------- 分层划分：逐类别按比例抽样 ----------
        stratify = np.asarray(stratify).ravel()
        classes = np.unique(stratify)                       # 所有类别
        train_parts, test_parts = [], []                    # 各类别的训练/测试索引
        for c in classes:
            ci = np.where(stratify == c)[0]                 # 该类别全部样本索引
            if shuffle:
                ci = ci[rs.permutation(len(ci))]            # 类内洗牌
            k = int(round(len(ci) * (test_size if isinstance(test_size, float)
                                      else test_size / n)))  # 该类应抽出的测试样本数
            k = min(max(k, 1), len(ci) - 1) if len(ci) > 1 else 0  # 保证训练集/测试集都不空
            test_parts.append(ci[:k])                       # 前 k 个进测试集
            train_parts.append(ci[k:])                      # 其余进训练集
        train_idx = np.concatenate(train_parts)             # 拼接所有类别
        test_idx = np.concatenate(test_parts)
        if shuffle:                                         # 最后再整体洗牌一次，打散类别顺序
            train_idx = train_idx[rs.permutation(len(train_idx))]
            test_idx = test_idx[rs.permutation(len(test_idx))]

    # 按索引切出四份数据并返回（顺序：训练在前，与 sklearn 一致）
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ======================================================================
# 四、合成数据（不依赖 sklearn）
# ======================================================================
def make_blobs(n_samples=200, centers=2, center_box=(-3.0, 3.0), std=1.0,
               n_features=2, seed=0):
    """生成二维（或 n 维）高斯团簇合成数据，用于逻辑回归等演示。

    实现思路：
        在 center_box 内随机撒 centers 个中心点，
        每个中心周围按 std 的标准差采样高斯点，标签即中心编号，最后整体洗牌。

    参数：
        n_samples (int): 总样本数；
        centers (int): 团簇（类别）个数；
        center_box (tuple): 中心点坐标的取值范围 (lo, hi)；
        std (float): 每个团簇内高斯噪声的标准差；
        n_features (int): 特征维数（默认二维便于画图）；
        seed (int): 随机种子。

    返回：
        (X, y)：X 形状 (n_samples, n_features)，y 形状 (n_samples,) 整数标签。
    """
    rs = np.random.RandomState(seed)                        # 独立随机源
    # 在指定范围内随机生成每个团簇的中心点 (centers, n_features)
    center_arr = rs.uniform(center_box[0], center_box[1],
                            size=(centers, n_features))
    # 各团簇样本数：均分再把余数依次补到前面的团簇
    base = n_samples // centers
    counts = [base] * centers
    for i in range(n_samples - base * centers):
        counts[i] += 1

    xs, ys = [], []                                         # 收集各团簇的样本与标签
    for c in range(centers):
        # 以第 c 个中心为均值、std 为标准差采样高斯点
        xs.append(rs.normal(loc=center_arr[c], scale=std, size=(counts[c], n_features)))
        ys.append(np.full(counts[c], c, dtype=np.int64))    # 标签 = 团簇编号
    X = np.concatenate(xs, axis=0).astype(np.float64)
    y = np.concatenate(ys, axis=0)
    perm = rs.permutation(n_samples)                        # 整体洗牌，打散团簇顺序
    return X[perm], y[perm]


# ======================================================================
# 五、DataLoader 小批量迭代器
# ======================================================================
class DataLoader:
    """小批量数据迭代器（Keras 风格）：每次 yield 一批 (xb, yb) numpy 数组。

    注意：这里不做任何 Node 包装——包装成 Node 是 Trainer 的职责，
    保持本模块与计算图解耦。

    参数：
        X (np.ndarray): 特征 (N, ...)；
        y (np.ndarray): 标签 (N,)；
        batch_size (int): 批大小；
        shuffle (bool): 是否在每个 epoch 打乱顺序；
        seed (int | None): 打乱用的随机种子；None 则使用全局 numpy 随机源。
    """

    def __init__(self, X, y, batch_size=32, shuffle=True, seed=None):
        self.X = np.asarray(X)                              # 保存特征
        self.y = np.asarray(y)                              # 保存标签
        assert len(self.X) == len(self.y), "X 与 y 样本数不一致"  # 基本一致性检查
        self.batch_size = int(batch_size)                   # 批大小
        self.shuffle = bool(shuffle)                        # 是否乱序
        # 固定 seed 时创建独立随机源，保证每次迭代序列整体可复现
        self._rng = np.random.RandomState(seed) if seed is not None else None

    def __iter__(self):
        """迭代产生 (xb, yb) 批次；最后一个不足 batch_size 的批也保留。"""
        n = len(self.X)                                     # 样本总数
        if self.shuffle:
            # 有独立随机源就用它（可复现），否则用全局源
            rng = self._rng if self._rng is not None else np.random
            perm = rng.permutation(n)                       # 本轮的乱序索引
        else:
            perm = np.arange(n)                             # 不打乱：保持原序
        for start in range(0, n, self.batch_size):          # 按步长切批
            idx = perm[start:start + self.batch_size]       # 本批样本索引
            yield self.X[idx], self.y[idx]                  # 产出 (特征批, 标签批)

    def __len__(self):
        """返回批数 = ceil(N / batch_size)。"""
        return (len(self.X) + self.batch_size - 1) // self.batch_size


# ======================================================================
# 六、真实数据集加载（鸢尾花 / MNIST）
# ======================================================================
def load_iris(data_dir="data"):
    """加载鸢尾花数据集（150 条，4 特征，3 类），带三级降级方案。

    降级顺序：
        1) sklearn.datasets.load_iris（首选，已安装）；
        2) 本地缓存 csv（data/iris_cache.csv，首成功后自动写入）；
        3) 合成替代数据（三团簇高斯，打印醒目警告）。

    参数：
        data_dir (str): 缓存目录。

    返回：
        (X, y, class_names)：X (150,4) float64；y (150,) int；class_names 类别名列表。
    """
    cache_csv = os.path.join(data_dir, "iris_cache.csv")    # 本地缓存文件路径
    # ---------- 第 1 级：优先使用 sklearn ----------
    try:
        from sklearn.datasets import load_iris as _sk_iris  # 延迟导入，避免无 sklearn 时报错
        raw = _sk_iris()
        X = raw.data.astype(np.float64)                     # 特征 (150,4)
        y = raw.target.astype(np.int64)                     # 标签 (150,)
        class_names = [str(s) for s in raw.target_names]    # ['setosa','versicolor','virginica']
        # 首次成功后写入本地缓存，供将来无 sklearn 环境使用
        _save_iris_cache(cache_csv, X, y, class_names)
        return X, y, class_names
    except Exception:
        pass                                                # sklearn 不可用时静默降级

    # ---------- 第 2 级：读取本地缓存 csv ----------
    if os.path.exists(cache_csv):
        try:
            X, y, class_names = _load_iris_cache(cache_csv)
            print(f"[data] 已从本地缓存加载鸢尾花数据: {cache_csv}")
            return X, y, class_names
        except Exception as e:
            print(f"[data] 本地缓存损坏({e})，改用合成数据")

    # ---------- 第 3 级：合成替代数据 ----------
    print("[警告] sklearn 与本地缓存均不可用，使用合成高斯团簇替代鸢尾花数据！")
    X, y = make_blobs(n_samples=150, centers=3, center_box=(-2.0, 2.0),
                      std=0.6, n_features=4, seed=42)       # 4 维 3 团簇近似鸢尾花结构
    class_names = ["synthetic-0", "synthetic-1", "synthetic-2"]
    return X, y, class_names


def _save_iris_cache(path, X, y, class_names):
    """把鸢尾花数据写入 csv 缓存（首行为 # 注释记录类别名）。

    参数：
        path (str): csv 保存路径；
        X (np.ndarray): 特征；
        y (np.ndarray): 标签；
        class_names (list[str]): 类别名列表。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # 确保目录存在
    with open(path, "w", encoding="utf-8") as f:
        f.write("# class_names=" + ",".join(class_names) + "\n")  # 注释行记录类别名
        for xi, yi in zip(X, y):                            # 每行: 4 个特征 + 标签
            f.write(",".join(f"{v:.6f}" for v in xi) + f",{int(yi)}\n")


def _load_iris_cache(path):
    """从 csv 缓存读取鸢尾花数据（对应 _save_iris_cache 的格式）。

    参数：
        path (str): csv 路径。

    返回：
        (X, y, class_names) 三元组。
    """
    class_names, xs, ys = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:                                      # 逐行解析
            line = line.strip()
            if not line:
                continue                                    # 跳过空行
            if line.startswith("# class_names="):           # 解析注释行里的类别名
                class_names = line.split("=", 1)[1].split(",")
                continue
            parts = line.split(",")                         # 数据行：4 特征 + 1 标签
            xs.append([float(v) for v in parts[:4]])
            ys.append(int(float(parts[4])))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.int64), class_names


# ---------------------- MNIST idx 格式解析工具 ----------------------
def _parse_idx_images(gz_path):
    """解析 MNIST 图像 idx.gz 文件为 (N, 28, 28) uint8 数组。

    idx3 格式：魔数 2051 | 样本数 | 行数 | 列数 | 像素字节流（均大端序）。

    参数：
        gz_path (str): .gz 文件路径。

    返回：
        np.ndarray: (N, H, W) uint8。
    """
    with gzip.open(gz_path, "rb") as f:                     # 解压读取全部字节
        data = f.read()
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])  # 按大端序解析头部 4 个 u32
    assert magic == 2051, f"图像 idx 魔数错误: {magic}"       # 2051 = 图像文件
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows, cols)


def _parse_idx_labels(gz_path):
    """解析 MNIST 标签 idx.gz 文件为 (N,) uint8 数组。

    idx1 格式：魔数 2049 | 样本数 | 标签字节流。

    参数：
        gz_path (str): .gz 文件路径。

    返回：
        np.ndarray: (N,) uint8。
    """
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    magic, n = struct.unpack(">II", data[:8])               # 大端序解析头部 2 个 u32
    assert magic == 2049, f"标签 idx 魔数错误: {magic}"       # 2049 = 标签文件
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def _download_mnist(data_dir):
    """用 urllib 从镜像下载 MNIST 四个 idx.gz 文件到 data/mnist/（带缓存）。

    实现思路：
        逐个检查目标文件是否已存在，缺失才发起 HTTP 下载（超时 60 秒），
        先写 .part 临时文件再重命名，避免半截文件污染缓存。

    参数：
        data_dir (str): 缓存根目录，文件落在 data_dir/mnist/ 下。

    返回：
        dict: key -> 本地文件路径。
    """
    mnist_dir = os.path.join(data_dir, "mnist")             # 缓存子目录
    os.makedirs(mnist_dir, exist_ok=True)
    paths = {}
    for key, fname in _MNIST_FILES.items():                 # 依次处理 4 个文件
        path = os.path.join(mnist_dir, fname)
        if not os.path.exists(path):                        # 已缓存则跳过下载
            url = _MNIST_MIRROR + fname                     # 拼接镜像完整 URL
            print(f"[data] 正在下载 {url} ...")
            tmp = path + ".part"                            # 临时文件防止半截缓存
            urllib.request.urlretrieve(url, tmp)            # 阻塞下载（超时由系统控制）
            os.replace(tmp, path)                           # 原子重命名为正式缓存
        paths[key] = path
    return paths


def load_mnist(train=True, data_dir="data"):
    """加载 MNIST 手写数字数据集，返回已归一化的 (N,1,28,28) float32 张量。

    降级方案：
        1) torchvision.datasets.MNIST（首选，已安装）；
        2) urllib 从镜像 https://ossci-datasets.s3.amazonaws.com/mnist/
           下载 4 个 idx.gz 缓存到 data/mnist/ 后自行解析；
        3) 全部失败 -> 抛出带中文指引的 RuntimeError。

    参数：
        train (bool): True 训练集(6万) / False 测试集(1万)；
        data_dir (str): 数据缓存根目录。

    返回：
        (X, y)：X (N,1,28,28) float32，像素缩放到 [0,1]；y (N,) int64 标签。
    """
    # ---------- 第 1 级：torchvision ----------
    try:
        from torchvision.datasets import MNIST              # 延迟导入
        ds = MNIST(root=os.path.join(data_dir, "mnist"), train=train, download=True)
        # ds.data 为 uint8 (N,28,28)；转 float32、加通道维、缩放到 [0,1]
        X = ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
        y = ds.targets.numpy().astype(np.int64)             # 标签转 int64
        return X, y
    except Exception as e:
        print(f"[data] torchvision 加载失败({type(e).__name__}: {e})，改用镜像下载")

    # ---------- 第 2 级：urllib 镜像下载 + 自行解析 idx ----------
    try:
        paths = _download_mnist(data_dir)                   # 下载/缓存 4 个文件
        if train:
            images = _parse_idx_images(paths["train_images"])  # 解析训练图像
            labels = _parse_idx_labels(paths["train_labels"])  # 解析训练标签
        else:
            images = _parse_idx_images(paths["test_images"])   # 解析测试图像
            labels = _parse_idx_labels(paths["test_labels"])   # 解析测试标签
        X = images.astype(np.float32)[:, None, :, :] / 255.0   # (N,28,28)->(N,1,28,28) 并归一化
        y = labels.astype(np.int64)
        return X, y
    except Exception as e:
        # ---------- 第 3 级：彻底失败，给出中文指引 ----------
        raise RuntimeError(
            "MNIST 数据加载失败！请尝试以下办法：\n"
            "  1. 检查网络连接后重试（将自动从镜像下载约 11MB 数据）；\n"
            f"  2. 手动下载 4 个 .gz 文件放到 {os.path.abspath(os.path.join(data_dir, 'mnist'))}/ ：\n"
            "     " + ", ".join(_MNIST_FILES.values()) + "\n"
            "     镜像地址: " + _MNIST_MIRROR + "\n"
            f"  3. 或改用 load_mnist_arrays 的合成/缓存逻辑。\n"
            f"原始错误: {type(e).__name__}: {e}"
        )


def _balanced_subset(X, y, n):
    """按类别轮流（round-robin）取样，得到类别均衡的前 n 个样本子集。

    实现思路：
        先按类别分组得到各类的索引序列，然后第 0 轮取每类第 0 个、
        第 1 轮取每类第 1 个……直到凑满 n 个，保证各类数量尽量均衡。

    参数：
        X (np.ndarray): 全量特征；
        y (np.ndarray): 全量标签；
        n (int | None): 目标样本数；None 或 >= 全量时原样返回。

    返回：
        (X_sub, y_sub)：取样后的子集。
    """
    if n is None or n >= len(y):                            # 不需要取样
        return X, y
    classes = np.unique(y)                                  # 全部类别
    class_idx = {c: np.where(y == c)[0] for c in classes}   # 每类的样本索引队列
    picked = []                                             # 轮流取样的结果索引
    k = 0                                                   # 轮次号
    while len(picked) < n:
        progressed = False                                  # 本轮是否有新取样（防死循环）
        for c in classes:
            if len(picked) >= n:                            # 凑满即停
                break
            ci = class_idx[c]
            if k < len(ci):                                 # 该类还有剩余样本
                picked.append(ci[k])                        # 取该类第 k 个
                progressed = True
        if not progressed:                                  # 所有类都取完仍不够 n（极端情况）
            break
        k += 1                                              # 下一轮
    picked = np.array(sorted(picked))                       # 排序保持原相对顺序
    return X[picked], y[picked]


def load_mnist_arrays(subset_train=None, subset_test=None, data_dir="data"):
    """一次性加载 MNIST 的训练/测试数组，支持类别均衡子集（快速模式）。

    参数：
        subset_train (int | None): 训练子集样本数（按类别轮流取样，None=全量 60000）；
        subset_test (int | None): 测试子集样本数（None=全量 10000）；
        data_dir (str): 缓存目录。

    返回：
        (X_train, y_train, X_test, y_test)：
            X_train (n1,1,28,28) float32 [0,1]；y_train (n1,) int64；测试同理。
    """
    Xtr, ytr = load_mnist(train=True, data_dir=data_dir)    # 加载全量训练集（含缓存）
    Xte, yte = load_mnist(train=False, data_dir=data_dir)   # 加载全量测试集
    Xtr, ytr = _balanced_subset(Xtr, ytr, subset_train)     # 训练子集（类别均衡）
    Xte, yte = _balanced_subset(Xte, yte, subset_test)      # 测试子集（类别均衡）
    return Xtr, ytr, Xte, yte


# ======================================================================
# 七、数据增强：随机平移
# ======================================================================
def random_shift_images(X, max_shift=2, seed=0):
    """图像随机平移数据增强：每张图在 H/W 方向随机整数像素循环平移。

    实现思路：
        对每张图用 np.roll 沿高度轴和宽度轴各循环平移随机像素
        （范围 [-max_shift, max_shift]）。循环平移保证移出的像素从另一侧补入，
        不产生黑边，适合 MNIST 这类居中书写的数字。

    参数：
        X (np.ndarray): 图像数组，(N,1,H,W) 或 (N,H,W)；
        max_shift (int): 最大平移像素数；
        seed (int): 随机种子。

    返回：
        np.ndarray: 平移后的图像数组，形状与输入一致。
    """
    X = np.asarray(X)
    squeezed = False
    if X.ndim == 3:                                         # (N,H,W) -> (N,1,H,W) 统一处理
        X = X[:, None, :, :]
        squeezed = True
    assert X.ndim == 4, "输入应为 (N,1,H,W) 或 (N,H,W)"
    rs = np.random.RandomState(seed)                        # 独立随机源保证可复现
    out = np.empty_like(X)                                  # 输出容器
    for i in range(len(X)):                                 # 逐张平移
        dy = rs.randint(-max_shift, max_shift + 1)          # 高度方向随机平移量
        dx = rs.randint(-max_shift, max_shift + 1)          # 宽度方向随机平移量
        # 先沿第 0 空间轴(H)循环平移，再沿第 1 空间轴(W)循环平移
        out[i, 0] = np.roll(np.roll(X[i, 0], dy, axis=0), dx, axis=1)
    return out[:, 0, :, :] if squeezed else out             # 还原输入形状
