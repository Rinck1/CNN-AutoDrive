# -*- coding: utf-8 -*-
"""
common_setup.py —— 项目公共环境初始化模块
====================================================

所有可运行脚本（applications/、experiments/ 下的脚本）开头都调用
``init_env()``，统一完成以下四件事，保证在任何 Windows 环境下行为一致：

1. 控制台强制 UTF-8 输出   —— 避免中文在 Windows 默认 GBK 控制台乱码/报错；
2. 固定全局随机种子        —— random + numpy，保证实验可复现；
3. matplotlib 中文显示     —— 指定微软雅黑/黑体等字体，负号正常显示；
4. 强制 Agg 后端           —— 无界面环境下也能 savefig 保存图片，禁止 plt.show()。

注意：本文件由项目负责人维护，其他脚本只允许 import，不允许修改。
"""

import os
import random
import sys

import numpy as np


def init_env(seed=42, output_dir=None):
    """初始化脚本运行环境（所有可运行脚本的入口处调用一次）。

    参数：
        seed (int): 全局随机种子，保证实验可复现；
        output_dir (str | None): 若给定则自动创建该输出目录（支持 outputs/exp1 这类子目录）。

    返回：
        str | None: 若传了 output_dir 则返回其绝对路径，否则返回 None。
    """
    # ---------- 1. Windows 控制台强制 UTF-8 输出，避免中文乱码 ----------
    for _stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(_stream, "reconfigure"):
                _stream.reconfigure(encoding="utf-8", errors="ignore")
        except Exception:
            pass  # 个别环境（如重定向到文件）可能不支持 reconfigure，直接忽略

    # ---------- 2. 固定随机种子，保证结果可复现 ----------
    random.seed(seed)
    np.random.seed(seed)

    # ---------- 3. matplotlib 中文字体 + 无界面后端 ----------
    import matplotlib
    matplotlib.use("Agg")  # 不弹窗，纯保存图片
    import matplotlib.pyplot as plt
    # Windows 自带微软雅黑/黑体，按优先级尝试；负号用 ASCII 渲染避免乱码
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                                       "KaiTi", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    # ---------- 4. 按需创建输出目录 ----------
    out_path = None
    if output_dir:
        out_path = os.path.abspath(output_dir)
        os.makedirs(out_path, exist_ok=True)

    return out_path
