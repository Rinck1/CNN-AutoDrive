# -*- coding: utf-8 -*-
"""
web_app.py —— MiniFlows 全功能交互式 Web 平台快捷启动入口
==========================================================

直接运行：
    python web_app.py [--port 8000]
浏览器访问：
    http://127.0.0.1:8000/
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from applications.web_studio import main

if __name__ == '__main__':
    main()
