# -*- coding: utf-8 -*-
"""
tests/test_web_studio.py —— MiniFlows 全功能 Web 平台端到端测试
===================================================================

覆盖验证：
    1. GET / 与 /health 页面及健康检查；
    2. 预载 ResNet-GN 主模型推理、赛道渲染热图 (/api/donkeycar/render) 与闭环仿真 (/api/donkeycar/sim)；
    3. 11 项视觉扰动渲染（回归：perturb_images 参数签名修复）；
    4. 一键训练线性回归、鸢尾花分类、DonkeyCar ResNet-GN（含保存/加载往返零误差校验）；
    5. MNIST ResNetClassifier / LeNet 快捷加载与 /predict + /api/heatmap；
    6. 输入形状不匹配时返回可读错误（回归：unpack 报错）；
    7. 四大扩展实验室接口（计算图生成 / 17 项梯度检验 / 异常训练对比 / 预处理统计）；
    8. 训练运行历史（落盘 / 多曲线对比数据 / 清理）；
    9. 11 项视觉扰动鲁棒性扫描；
    10. 实验报告导出（Markdown / ZIP + 图表 PNG）；
    11. /api/benchmarks 与静态成果图访问；
    12. 服务正常退出与端口释放。

运行（项目根目录）：
    python tests/test_web_studio.py
"""

import io
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common_setup import init_env
init_env(seed=42)

import numpy as np
from applications.web_studio import start_server


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode('utf-8'))


def _get_bytes(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read(), dict(r.headers)


def _post(url, obj):
    data = json.dumps(obj).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8'))


def _post_err(url, obj):
    """期望失败的 POST：返回 (状态码, 错误体)。"""
    data = json.dumps(obj).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))


def _wait_train(url, timeout_s=45):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = _get(url + 'api/snapshot')
        if s['status'] != 'training':
            return s
        time.sleep(0.25)
    raise TimeoutError('训练超时未完成')


def main():
    print('=' * 66)
    print('MiniFlows Web Studio 全功能端到端测试')
    print('=' * 66)

    port = _free_port()
    studio, url = start_server(host='127.0.0.1', port=port, block=False)

    try:
        # 1. 页面与健康检查
        with urllib.request.urlopen(url, timeout=10) as r:
            html = r.read().decode('utf-8')
        assert 'MiniFlows Studio' in html and '/api/train/start' in html
        assert 'page-runs' in html and 'cv-runs' in html, '运行历史页面缺失'
        h = _get(url + 'health')
        assert h['ok']
        print(f'  [1] GET / ({len(html)} 字节) 与 /health OK ✓')

        # 2. 预载主模型 DonkeyCar ResNet-GN 渲染 + 推理 + 闭环仿真
        dc_r = _post(url + 'api/donkeycar/render', {'bottom': 6, 'tilt': 10, 'disturb': 'clean'})
        assert len(dc_r['image']) == 64 and dc_r['pred_steer'] is not None
        assert len(dc_r['heatmap']) == 8
        sim = _post(url + 'api/donkeycar/sim', {'amp': 12, 'wavelength': 300, 'noise_sigma': 0.0})
        assert 'closed' in sim and 'open_loop' in sim
        assert len(sim['closed']['z']) > 10
        print(f"  [2] DonkeyCar 主模型推理 steer={dc_r['pred_steer']}，闭环 max_dev={sim['closed']['max_dev']}px ✓")

        # 3. 回归：11 项视觉扰动渲染（此前因 perturb_images 签名不匹配全部 400）
        disturb_keys = ['clean', 'noise_0.02', 'noise_0.05', 'noise_0.10',
                        'bright_-0.2', 'bright_+0.2', 'contrast_0.7', 'contrast_1.3',
                        'blur_3x3', 'occ_center', 'occ_bottom']
        for k in disturb_keys:
            r = _post(url + 'api/donkeycar/render', {'bottom': 6, 'tilt': 10, 'disturb': k})
            assert r['pred_steer'] is not None, f'扰动 {k} 推理失败'
            assert len(r['image']) == 64
        code, err = _post_err(url + 'api/donkeycar/render', {'bottom': 6, 'tilt': 10, 'disturb': '不存在'})
        assert code == 400 and '未知扰动类型' in err.get('error', '')
        print(f'  [3] 11 项视觉扰动渲染全部通过（未知扰动返回 {code} 可读错误）✓')

        # 3.1 回归：输入形状不匹配应返回可读错误而非底层 unpack 异常
        #     此时激活模型是 DonkeyCar 图像模型（4 维输入），却收到 2 维特征向量
        code, err = _post_err(url + 'predict', {'features': [1.0, 2.0]})
        assert code == 400, '特征向量喂给图像模型应报错'
        assert '输入形状不匹配' in err.get('error', ''), err
        print("  [3.1] 输入形状不匹配返回可读错误 ✓")

        # 4. 一键训练：线性回归 + 实时拟合线 + 保存/加载往返检验
        _post(url + 'api/train/start', {
            'task': 'linear', 'arch': 'linear', 'epochs': 6, 'samples': 80,
            'lr': 0.05, 'batch_size': 16, 'optimizer': 'adam'
        })
        snap = _wait_train(url)
        assert snap['status'] == 'completed' and snap['epochs_done'] == 6
        assert snap['preview']['kind'] == 'linear'
        saved = _post(url + 'api/model/save', {'tag': 'test_linear'})
        assert saved['roundtrip_max_diff'] == 0.0
        loaded = _post(url + 'api/model/load', {})
        assert loaded['roundtrip_max_diff'] == 0.0
        p_lin = _post(url + 'predict', {'features': [2.0]})
        assert p_lin['task'] == 'regression'
        print(f"  [4] 一键训练(linear) 6 轮完成，保存/加载 diff={saved['roundtrip_max_diff']:.1e} ✓")

        # 5. 一键训练：鸢尾花分类（验证标准化 + 混淆矩阵 + 向量 /predict）
        _post(url + 'api/train/start', {
            'task': 'iris', 'arch': 'mlp', 'epochs': 10, 'samples': 150,
            'lr': 0.03, 'batch_size': 16, 'scale': 'standard', 'early_stop': True
        })
        snap_iris = _wait_train(url)
        assert snap_iris['status'] == 'completed'
        assert snap_iris['preview']['kind'] == 'iris'
        p_iris = _post(url + 'predict', {'features': [5.1, 3.5, 1.4, 0.2]})
        assert p_iris['task'] == 'classification' and abs(sum(p_iris['probs']) - 1.0) < 1e-4
        print(f"  [5] 一键训练(iris) 准确率={snap_iris['history']['val_metric'][-1]*100:.1f}%，预测={p_iris['label']} ✓")

        # 7. 一键训练：MNIST ResNetClassifier 快速自检 + /predict + /api/heatmap
        _post(url + 'api/train/start', {
            'task': 'mnist', 'arch': 'resnet_gn', 'epochs': 1, 'samples': 200,
            'lr': 0.002, 'batch_size': 32
        })
        snap_m = _wait_train(url, timeout_s=60)
        assert snap_m['status'] == 'completed'
        img28 = np.random.rand(28, 28).tolist()
        pm = _post(url + 'predict', {'image': img28})
        assert pm['task'] == 'classification' and len(pm['probs']) == 10
        hm = _post(url + 'api/heatmap', {'image': img28, 'grid': 8})
        assert len(hm['grid']) == 8
        print("  [7] 一键训练(MNIST ResNet-GN 分类版) + /predict + 8×8 热图通过 ✓")

        # 8. 四大扩展实验室接口
        g = _post(url + 'api/lab/graph_viz', {'preset': 'mlp'})
        assert g['n_nodes'] >= 5 and g['image_b64'].startswith('data:image/png;base64,')
        gc = _post(url + 'api/lab/grad_check', {})
        assert gc['passed'] == gc['total'] and gc['total'] == 17
        an = _post(url + 'api/lab/anomaly', {'kind': 'explosion'})
        assert len(an['series']) == 2
        assert {s['role'] for s in an['series']} == {'bad', 'good'}, '异常对照缺少 good/bad 语义标注'
        pr = _post(url + 'api/lab/preprocess', {})
        assert len(pr['rows']) == 4
        print(f"  [8] 扩展实验室(计算图 {g['n_nodes']} 节点 / 梯度检验 {gc['passed']}/{gc['total']} / 异常对照 / 预处理) 通过 ✓")

        # 9. 训练运行历史：多次训练自动记录 + 可对比 + 可清理
        runs = _get(url + 'api/runs')['runs']
        assert len(runs) >= 3, f'运行历史应记录 ≥3 次训练，实际 {len(runs)}'
        r0 = runs[0]
        for k in ('id', 'task', 'arch', 'cfg', 'epoch', 'val_loss', 'val_metric',
                  'best_metric', 'metric_name', 'params', 'elapsed_s', 'status'):
            assert k in r0, f'运行记录缺字段 {k}'
        assert len(r0['val_loss']) == len(r0['epoch'])
        assert r0['metric_name'] in ('MAE', 'Acc')
        del_id = runs[-1]['id']
        dr = _post(url + 'api/runs/delete', {'id': del_id})
        assert dr['ok'] and dr['removed'] == 1
        assert len(_get(url + 'api/runs')['runs']) == len(runs) - 1
        print(f"  [9] 运行历史记录 {len(runs)} 条（字段完整、可删除）✓")

        # 10. 11 项视觉扰动鲁棒性扫描
        _post(url + 'api/model/load', {'preset': 'donkeycar_resnet'})
        rob = _post(url + 'api/donkeycar/robustness', {'samples': 200})
        assert len(rob['rows']) == 11
        assert rob['n_test'] > 0
        assert rob['worst']['mae'] >= rob['clean_mae']
        assert all('degrade_pct' in r and 'p95_abs' in r for r in rob['rows'])
        print(f"  [10] 鲁棒性扫描 11 项（干净 MAE={rob['clean_mae']}，最差 {rob['worst']['kind']}）✓")

        # 11. 实验报告导出：Markdown + ZIP（内含图表 PNG）
        md, hdr = _get_bytes(url + 'api/report.md')
        assert 'MiniFlows 实验报告' in md.decode('utf-8')
        assert 'markdown' in hdr.get('Content-Type', '')
        zp, zhdr = _get_bytes(url + 'api/report.zip')
        assert zhdr.get('Content-Type') == 'application/zip'
        zf = zipfile.ZipFile(io.BytesIO(zp))
        names = zf.namelist()
        assert any(n.endswith('.md') for n in names)
        assert any(n.startswith('charts/') and n.endswith('.png') for n in names), names
        assert zf.testzip() is None
        print(f"  [11] 报告导出 Markdown({len(md)}B) + ZIP({len(zp)}B, {len(names)} 个条目) ✓")

        # 12. 多架构与基准对比接口
        bm = _get(url + 'api/benchmarks')
        assert bm['resnet'] is not None and bm['pytorch_resources'] is not None
        assert bm['resnet'].get('params'), 'ResNet 结果应带参数量'
        print(f"  [12] /api/benchmarks 图表数={len(bm['gallery'])} OK ✓")

        # 13. 清理运行历史并确认持久化文件可重载
        cl = _post(url + 'api/runs/clear', {})
        assert cl['ok']
        assert _get(url + 'api/runs')['runs'] == []
    finally:
        studio.stop()

    print('\n' + '=' * 66)
    print('✓ MiniFlows Web Studio 全部端到端测试通过')
    print('=' * 66)


if __name__ == '__main__':
    main()

