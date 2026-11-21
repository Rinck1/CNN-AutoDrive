# -*- coding: utf-8 -*-
"""
applications/inspector.py —— 模型自省：结构追踪 / 反向计算图 / 3D 布局
=============================================================================

把 MiniFlows 的模型变成「可被前端画出来」的数据：

1. **结构追踪**（trace_architecture）
   跑一次前向，逐层记录输入/输出形状、参数量、层类型，
   供前端做「一层层拆开」的 3D 架构动画与逐层说明。

2. **反向计算图**（trace_backward）
   跑一次前向 + 反向，收集计算图的节点（算子名 / 形状 / 梯度 L2 范数）
   与边（父 → 子），供前端做反向传播的流动动画。

3. **3D 布局**（layer_layout）
   把层按数据流方向排在 X 轴上，用输出张量的规模决定 Y/Z 方向的体量，
   权重张量本身给出点云坐标，供「3D 参数可视化」使用。

本模块只依赖 numpy 与 miniflows 自身，不引入任何第三方依赖。
"""

import math
import types

import numpy as np

from miniflows.graph import Node
from miniflows.layers import Sequential, Layer
from miniflows.utils import count_params


# ======================================================================
# 一、层类型 → 语义分类（前端按此决定颜色 / 图标 / 说明文案）
# ======================================================================
_LAYER_KIND = {
    'Linear': 'linear',
    'Conv2D': 'conv',
    'MaxPool2D': 'pool',
    'AvgPool2D': 'pool',
    'Flatten': 'flatten',
    'ReLU': 'act',
    'Sigmoid': 'act',
    'Tanh': 'act',
    'Softmax': 'act',
    'GELU': 'act',
    'Dropout': 'dropout',
    'GroupNorm2D': 'norm',
    'LayerNorm': 'norm',
    'MultiHeadSelfAttention': 'attn',
    'TransformerBlock': 'block',
    'PatchEmbedding': 'embed',
    'BasicBlock': 'block',
    'Sequential': 'seq',
    'ResNetRegressor': 'model',
    'ViTRegressor': 'model',
}


def _kind_of(layer):
    """返回层的语义分类。"""
    return _LAYER_KIND.get(type(layer).__name__, 'other')


# ======================================================================
# 二、结构追踪
# ======================================================================
def _child_layers(layer):
    """找出容器层持有的子层，返回 [(在父级命名方案下的名字, 子层)]。

    命名规则与各模型的 named_params 保持一致，这样层路径能和权重名对上：
      - Sequential 容器：用子层自身的 name，缺省 layer{i}（i 为容器内序号）；
      - 自定义主干（ResNetRegressor / BasicBlock / ViTRegressor 等）：
        用属性名（stem / stage{i} / head / conv1 / gn1 / shortcut ...）。
    """
    if isinstance(layer, Sequential):
        return [(getattr(s, 'name', None) or f'layer{i}', s)
                for i, s in enumerate(layer.layers)]
    out = []
    if isinstance(layer, Layer):
        for attr, val in vars(layer).items():
            if isinstance(val, Layer):
                out.append((getattr(val, 'name', None) or attr, val))
            elif isinstance(val, (list, tuple)):
                for i, sub in enumerate(val):
                    if isinstance(sub, Layer):
                        out.append((getattr(sub, 'name', None) or f'{attr}{i}', sub))
    return out


def collect_leaves(model):
    """递归收集模型中的叶子层，返回 [(路径名, 层实例)]。

    路径名与 named_params 一致（如「stage0.block0.conv1」对应权重
    「stage0.block0.conv1.W」），根容器自身不参与前缀。
    """
    leaves = []

    def walk(layer, path):
        kids = _child_layers(layer)
        if kids:
            for cname, sub in kids:
                walk(sub, f'{path}.{cname}' if path else cname)
        else:
            leaves.append((path or type(layer).__name__, layer))

    walk(model, '')
    return leaves


def trace_architecture(model, probe=None, input_shape=(1, 64, 64)):
    """跑一次前向，逐层记录结构信息。

    参数：
        model: 待分析的模型；
        probe: 前向用的样本张量；None 时按 input_shape 造一张全零图。

    返回：
        dict: {'layers': [...], 'params': int, 'input_shape': [...], 'output_shape': [...]}
        每层包含 name / type / kind / in / out / params / 说明文案。
    """
    if probe is None:
        probe = np.zeros((1,) + tuple(input_shape), dtype=np.float64)
    probe = np.asarray(probe, dtype=np.float64)
    if probe.ndim == len(input_shape):
        probe = probe[None, ...]

    records = []
    leaves = collect_leaves(model)

    # 给每个叶子层的 forward 打补丁，记录其输入/输出形状
    patched = []
    for path, layer in leaves:
        rec = {'name': path, 'type': type(layer).__name__, 'kind': _kind_of(layer),
               'params': int(sum(p.data.size for p in layer.params())), 'in': None, 'out': None}
        records.append(rec)
        orig = layer.forward

        def make_forward(orig, rec):
            def forward(self, x):
                out = orig(x)
                rec['in'] = list(np.shape(x.data))
                rec['out'] = list(np.shape(out.data))
                rec['thumb'] = _activation_thumb(out.data)
                return out
            return forward

        layer.forward = types.MethodType(make_forward(orig, rec), layer)
        patched.append((layer, orig))

    model.train_mode(False)
    try:
        out = model(Node(probe))
        out_shape = list(np.shape(out.data))
    finally:
        for layer, orig in patched:
            layer.forward = orig

    # 顶层容器（Sequential 之外）若自身带参数，补一条记录
    top_params = count_params(model)
    inner = sum(r['params'] for r in records)
    if top_params > inner:
        records.append({
            'name': getattr(model, 'name', None) or type(model).__name__,
            'type': type(model).__name__, 'kind': _kind_of(model),
            'params': top_params - inner, 'in': None, 'out': out_shape,
        })

    for i, r in enumerate(records):
        r['idx'] = i
        r['desc'] = _describe(r)

    return {
        'layers': records,
        'params': top_params,
        'input_shape': list(probe.shape),
        'output_shape': out_shape,
        'depth': len(records),
    }


def _activation_thumb(arr, max_side=12):
    """把一层输出张量压成 max_side×max_side 的激活热力图数据。

    - 4D 特征图 (N,C,H,W)：对通道取 |x| 均值，得到 H×W 响应图；
    - 2D / 展平向量：重排成近似方形；
    - 逐层 99 分位归一化，保证不同量级的层都能看清纹理。
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 0 or a.size == 0:
        return None
    if a.ndim >= 4:
        b = a.reshape(a.shape[0], a.shape[1], -1)
        m = np.abs(b).mean(axis=1)[0]
        h, w = int(a.shape[-2]), int(a.shape[-1])
        m = m.reshape(h, w)
    elif a.ndim == 3:
        m = np.abs(a[0])
    elif a.ndim == 2:
        v = np.abs(a[0]).reshape(-1)
        side = int(math.ceil(math.sqrt(v.size)))
        m = np.zeros(side * side, dtype=np.float64)
        m[:v.size] = v
        m = m.reshape(side, side)
    else:
        v = np.abs(a).reshape(-1)
        side = int(math.ceil(math.sqrt(v.size)))
        m = np.zeros(side * side, dtype=np.float64)
        m[:v.size] = v
        m = m.reshape(side, side)

    hi = float(np.percentile(m, 99)) if m.size else 0.0
    if hi <= 1e-12:
        hi = float(m.max()) if m.size else 1.0
    if hi <= 1e-12:
        hi = 1.0
    m = np.clip(m / hi, 0.0, 1.0)

    h, w = m.shape
    if max(h, w) > max_side:
        th = max(1, h // max_side)
        tw = max(1, w // max_side)
        m = m[:th * max_side, :tw * max_side].reshape(max_side, th, max_side, tw).mean(axis=(1, 3))
    return {
        'shape': [int(m.shape[0]), int(m.shape[1])],
        'data': [[round(float(v), 3) for v in row] for row in m],
    }


def _describe(rec):
    """给每层生成一句中文说明。"""
    kind = rec['kind']
    t = rec['type']
    if kind == 'conv':
        k = rec.get('out') or []
        return f'卷积层：把 {rec.get("in")} 映射到 {rec.get("out")}，感受野在此扩张'
    if kind == 'pool':
        return f'池化层：{rec.get("in")} → {rec.get("out")}，下采样降分辨率'
    if kind == 'act':
        return f'激活函数 {t}：逐元素非线性，梯度的主要来源之一'
    if kind == 'norm':
        return f'归一化层 {t}：稳定分布，不依赖 batch 统计量'
    if kind == 'linear':
        return f'全连接层：{rec.get("in")} → {rec.get("out")}，承担通道间的信息聚合'
    if kind == 'flatten':
        return f'摊平层：{rec.get("in")} → {rec.get("out")}，把空间维折进特征维'
    if kind == 'attn':
        return f'自注意力层 {t}：{rec.get("out")}，在 token 之间做加权聚合'
    if kind == 'block':
        return f'{t}：{rec.get("out")}，组合多个子层'
    if kind == 'embed':
        return f'Patch 嵌入：{rec.get("in")} → {rec.get("out")}，图像切成 patch 后线性投影'
    if kind == 'model':
        return f'模型主干：{rec.get("out")}'
    return f'{t}：{rec.get("out")}'


# ======================================================================
# 三、反向计算图追踪
# ======================================================================
def trace_backward(model, loss_fn, target, probe):
    """跑一次前向 + 反向，收集计算图结构与各节点梯度范数。

    参数：
        model: 待分析的模型；
        loss_fn: 损失对象（MSELoss / CrossEntropyLoss ...）；
        target: 真实标签；
        probe: 单样本输入（batch=1，保证图规模可读）。

    返回：
        dict: {'nodes': [...], 'edges': [...], 'loss': float, 'layers': [...]}
    """
    probe = np.asarray(probe, dtype=np.float64)
    if probe.ndim == 3:
        probe = probe[None, ...]

    model.train_mode(False)
    model.zero_grad()
    x_node = Node(probe, requires_grad=False, name='input')
    # 参数节点自身只带短名（如 conv1.W），这里映射回 named_params 的完整路径，
    # 便于前端按 stage/block 聚合梯度。
    param_names = {id(p): nm for nm, p in model.named_params()}
    pred = model(x_node)
    loss = loss_fn(pred, target)
    loss_value = float(loss.data)
    loss.backward()

    # 收集需要梯度的子图（与 backward 的剪枝规则一致）
    seen = {}
    order = []
    stack = [(loss, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if id(node) in seen or not node.requires_grad:
            continue
        seen[id(node)] = node
        stack.append((node, True))
        for p in node.parents:
            stack.append((p, False))

    # 深度（离输入的距离）用于 3D 分层
    depth = {}

    def compute_depth(n, d=0):
        if id(n) in depth:
            return depth[id(n)]
        depth[id(n)] = d
        best = d
        for p in n.parents:
            best = max(best, compute_depth(p, d + 1) + 1)
        depth[id(n)] = best
        return best

    compute_depth(loss)

    nodes, index = [], {}
    for i, n in enumerate(order):
        idx = len(nodes)
        index[id(n)] = idx
        g = n.grad
        grad_norm = float(np.linalg.norm(g)) if g is not None else None
        shape = list(np.shape(n.data))
        nodes.append({
            'i': idx,
            'op': n.op_name,
            'name': param_names.get(id(n), n.name or ''),
            'shape': shape,
            'size': int(np.prod(shape)) if shape else 1,
            'grad': grad_norm,
            'depth': depth.get(id(n), 0),
            'is_param': n.op_name == 'Parameter',
        })

    edges = []
    for i, n in enumerate(order):
        for p in n.parents:
            j = index.get(id(p))
            if j is not None:
                edges.append({'s': j, 't': i, 'op': n.op_name})

    # 按「层」聚合：把参数节点按名字前缀归组，前端可逐层高亮
    layer_groups = {}
    for nd in nodes:
        nm = nd['name']
        if not nm:
            continue
        key = nm.rsplit('.', 1)[0] if '.' in nm else nm
        g = layer_groups.setdefault(key, {'name': key, 'grad': 0.0, 'count': 0})
        g['grad'] += float(nd['grad'] or 0.0)
        g['count'] += 1
    layers = sorted(layer_groups.values(), key=lambda g: -g['grad'])

    max_grad = max([n['grad'] for n in nodes if n['grad'] is not None] or [1.0])
    for nd in nodes:
        nd['grad_ratio'] = (nd['grad'] / max_grad) if (nd['grad'] is not None and max_grad > 0) else 0.0

    return {
        'nodes': nodes,
        'edges': edges,
        'layers': layers,
        'loss': loss_value,
        'max_grad': max_grad,
        'n_nodes': len(nodes),
        'n_edges': len(edges),
    }


# ======================================================================
# 四、3D 布局与权重点云
# ======================================================================
def _matrix_dims(shape):
    """把张量形状解析成 (channels, rows, cols)，用于 3D 矩阵网格示意。

    - (N,C,H,W) → (C,H,W)
    - (N,T,D)   → (1,T,D)
    - (N,F)     → (1,1,F)
    - (N,)      → (1,1,1)
    """
    shape = [int(s) for s in (shape or []) if s is not None]
    if len(shape) >= 4:
        return (max(1, shape[-3]), max(1, shape[-2]), max(1, shape[-1]))
    if len(shape) == 3:
        return (1, max(1, shape[-2]), max(1, shape[-1]))
    if len(shape) == 2:
        return (1, 1, max(1, shape[-1]))
    if len(shape) == 1:
        return (1, 1, max(1, shape[0]))
    return (1, 1, 1)


def layer_layout(layers, max_points_per_layer=600):
    """把层列表排成 3D「张量网络」布局。

    关键点：每层块沿数据流方向（X 轴）的厚度 w 由参数量决定，
    高度 h 由特征图空间尺寸决定，深度 d 由通道数决定；层间距按
    「两个相邻块的真实半宽 + 固定间隙」累积计算，保证块与块永不重叠
    （旧实现用固定等距 + 最大 2.6 的块宽，层一多就糊成一团）。

    返回：
        dict: {'layers': [...], 'span': float, 'max_h': float, 'max_d': float}
        每层附带 channels / rows / cols，供前端绘制矩阵网格示意。
    """
    n = max(1, len(layers))
    out = []
    sizes = []
    for i, rec in enumerate(layers):
        out_shape = rec.get('out') or rec.get('in') or []
        ch, rows, cols = _matrix_dims(out_shape)
        params = max(0, int(rec.get('params', 0) or 0))
        if len(out_shape) >= 4:
            # 卷积 / 池化特征图：宽高按 H/W 真实比例，厚度按通道数
            vh = float(np.clip(0.80 + rows * 0.090, 0.80, 4.80))
            vw = float(np.clip(0.80 + cols * 0.090, 0.80, 4.80))
            thick = float(np.clip(0.30 + math.log2(max(1, ch)) * 0.24, 0.30, 2.20))
        else:
            # 全连接 / 向量：画成窄高的向量卡片
            feats = max(1, cols)
            vh = float(np.clip(0.90 + math.log2(feats) * 0.10, 0.90, 2.20))
            vw = 1.15
            thick = 0.35
            rows = int(min(12, max(1, round(math.sqrt(feats)))))
            cols = int(min(12, max(1, math.ceil(feats / rows))))
        sizes.append((vw, vh, thick, ch, rows, cols))
        out.append({
            'idx': rec.get('idx', i),
            'name': rec.get('name', f'layer{i}'),
            'type': rec.get('type', ''),
            'kind': rec.get('kind', 'other'),
            'params': params,
            'y': 0.0,
            'z': 0.0,
            'w': round(thick, 4),
            'h': round(vh, 4),
            'd': round(vw, 4),
            'thick': round(thick, 4),
            'vw': round(vw, 4),
            'vh': round(vh, 4),
            'channels': int(ch),
            'rows': int(rows),
            'cols': int(cols),
            'vol': float(np.prod(rec.get('out') or [1])),
            'in_shape': rec.get('in'),
            'out_shape': rec.get('out'),
            'thumb': rec.get('thumb'),
            'desc': rec.get('desc', ''),
        })

    # 累积排布：相邻张量卡片的厚度和 + 间隙
    xs = [0.0]
    for i in range(1, n):
        pw, ph, pd, _, _, _ = sizes[i - 1]
        w, h, d, _, _, _ = sizes[i]
        gap = 0.75 + max(pd, d) * 0.18
        xs.append(xs[-1] + pw / 2.0 + gap + w / 2.0)

    mid = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
    for i, rec in enumerate(out):
        rec['x'] = round(float(xs[i] - mid), 4)

    span = (xs[-1] - xs[0]) + (sizes[-1][2] if sizes else 1.0)
    return {
        'layers': out,
        'span': round(float(max(6.0, span)), 4),
        'max_h': round(float(max((s[1] for s in sizes), default=1.0)), 4),
        'max_d': round(float(max((s[0] for s in sizes), default=1.0)), 4),
    }


def weight_pointcloud(named_params, max_points_per_layer=600):
    """把权重张量降采样成 3D 点云（坐标 + 数值），供 3D 参数可视化。

    参数：
        named_params: [(名字, Node), ...]；
        max_points_per_layer: 每个张量最多取多少个点。

    返回：
        list: [{'name', 'shape', 'total', 'points': [{'x','y','z','v','a'}]}]
    """
    out = []
    for name, p in named_params:
        arr = np.asarray(p.data, dtype=np.float64)
        if arr.ndim == 0 or arr.size == 0:
            continue
        flat = arr.reshape(-1)
        total = int(flat.size)
        # 等距抽样，最多 max_points_per_layer 个
        step = max(1, total // max_points_per_layer)
        sel = flat[::step][:max_points_per_layer]
        amax = float(np.max(np.abs(sel))) or 1.0
        # 坐标：把张量每个轴映射到 [-1,1]
        coords = _tensor_coords(arr.shape, len(sel))
        pts = []
        for i, v in enumerate(sel):
            c = coords[i]
            pts.append({
                'x': round(float(c[0]), 4),
                'y': round(float(c[1]), 4),
                'z': round(float(c[2]), 4),
                'v': round(float(v), 5),
                'a': round(float(abs(v) / amax), 4),
            })
        out.append({'name': name, 'shape': list(arr.shape), 'total': total,
                    'std': round(float(arr.std()), 5), 'points': pts})
    return out


def _tensor_coords(shape, count):
    """为一个 n 维张量生成 count 个三维坐标（超出的轴折叠进 Z）。"""
    shape = [max(1, int(s)) for s in shape]
    n = len(shape)
    coords = []
    total = int(np.prod(shape))
    # 用线性索引反解各轴坐标，再把多余轴折叠进 Z
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.insert(0, acc)
        acc *= s
    for i in range(count):
        lin = (i * max(1, total // max(1, count))) % total
        c = []
        for ax in range(n):
            c.append(((lin // strides[ax]) % shape[ax]) / max(1, shape[ax] - 1) * 2 - 1
                      if shape[ax] > 1 else 0.0)
        while len(c) < 3:
            c.append(0.0)
        coords.append((c[0], c[1], sum(c[3:]) / max(1, len(c[3:])) if len(c) > 3 else c[2]))
    return coords
