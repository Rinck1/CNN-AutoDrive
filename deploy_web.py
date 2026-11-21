# -*- coding: utf-8 -*-
"""
deploy_web.py —— MiniFlows Web Studio 一键部署工具
==================================================

双击 `一键部署Web.bat`（Windows）或在项目根目录运行：

    python deploy_web.py                # 127.0.0.1:8000，自动打开浏览器
    python deploy_web.py --lan          # 0.0.0.0，允许同一局域网访问
    python deploy_web.py --port 9000    # 指定端口（被占用时自动顺延）
    python deploy_web.py --no-open      # 只启动服务，不打开浏览器
    python deploy_web.py --check        # 仅检查依赖，不启动服务

做的事情：
    1. 检查 Python 版本与 numpy / matplotlib 依赖；
    2. 端口被占用时自动寻找下一个可用端口；
    3. 启动 MiniFlows Web Studio（后台线程 + 健康检查）；
    4. 打印访问地址（局域网模式下额外打印本机 IP）；
    5. Ctrl+C 安全退出并释放端口。
"""

import argparse
import os
import socket
import sys
import time
import urllib.request
import webbrowser

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass

_ROOT = os.path.dirname(os.path.abspath(__file__))
if getattr(sys, 'frozen', False):
    _ROOT = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def check_deps():
    """检查运行 Web 平台所需的最小依赖。返回缺失包列表。"""
    missing = []
    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append('numpy')
    try:
        import matplotlib  # noqa: F401
    except Exception:
        missing.append('matplotlib')
    return missing


def port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, int(port)))
            return True
        except OSError:
            return False


def pick_port(host, port, tries=30):
    """从指定端口起向后寻找第一个可用端口。"""
    for p in range(int(port), int(port) + tries):
        if port_free(host, p):
            return p
    raise RuntimeError(f'端口 {port}~{port + tries - 1} 都被占用，请用 --port 指定其他端口')


def lan_ip():
    """获取本机在局域网中的 IPv4 地址（失败时回退 127.0.0.1）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


def wait_healthy(url, timeout=20.0):
    import json
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + 'health', timeout=1.0) as r:
                if json.loads(r.read().decode('utf-8')).get('ok'):
                    return True
        except Exception:
            pass
        time.sleep(0.15)
    return False


def main():
    parser = argparse.ArgumentParser(description='MiniFlows Web Studio 一键部署')
    parser.add_argument('--host', default=None, help='监听地址（默认 127.0.0.1）')
    parser.add_argument('--port', type=int, default=8000, help='起始端口（默认 8000）')
    parser.add_argument('--lan', action='store_true', help='监听 0.0.0.0，允许局域网访问')
    parser.add_argument('--no-open', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--check', action='store_true', help='只检查依赖后退出')
    args = parser.parse_args()

    if sys.version_info < (3, 8):
        print(f'[部署] 需要 Python 3.8+，当前 {sys.version.split()[0]}')
        return 1

    missing = check_deps()
    if missing:
        print('[部署] 缺少依赖：' + ', '.join(missing))
        print('       请先运行： python -m pip install -r requirements.txt')
        return 1
    print('[部署] 依赖检查通过（numpy / matplotlib）。')
    if args.check:
        return 0

    host = args.host or ('0.0.0.0' if args.lan else '127.0.0.1')
    try:
        port = pick_port(host, args.port)
    except RuntimeError as exc:
        print(f'[部署] {exc}')
        return 1
    if port != args.port:
        print(f'[部署] 端口 {args.port} 已被占用，自动改用 {port}。')

    # 延迟导入，确保依赖检查先给出可读提示
    from applications.web_studio import start_server

    try:
        studio, url = start_server(host=host, port=port, block=False)
    except OSError as exc:
        print(f'[部署] 启动失败：{exc}')
        return 1

    if not wait_healthy(url):
        print(f'[部署] 服务健康检查超时：{url}')
        studio.stop()
        return 1

    print('')
    print('=' * 62)
    print('  MiniFlows Web Studio 已部署完成')
    print('=' * 62)
    print(f'  本机访问： {url}')
    if host in ('0.0.0.0', '::'):
        print(f'  局域网访问： http://{lan_ip()}:{port}/')
    print('  按 Ctrl+C 停止服务并释放端口')
    print('=' * 62)

    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('\n[部署] 正在停止服务...')
    finally:
        studio.stop()
    print('[部署] 已退出，端口已释放。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
