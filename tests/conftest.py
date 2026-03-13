#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest 配置文件
设置测试环境和路径
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 添加 foundationpose_msgs 到 Python 路径
msgs_install_path = os.path.join(
    project_root,
    "foundationpose_msgs",
    "install",
    "foundationpose_msgs",
    "lib",
    "python3.10",
    "site-packages",
)
if os.path.exists(msgs_install_path):
    sys.path.insert(0, msgs_install_path)
