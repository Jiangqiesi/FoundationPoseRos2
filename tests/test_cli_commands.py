#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 foundationpose_cli 命令行接口
"""

import pytest
import subprocess
import sys
import os


# CLI 脚本路径
CLI_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "foundationpose_cli.py"
)


class TestCLIHelp:
    """测试 CLI 帮助信息"""

    def test_main_help(self):
        """测试主命令 --help"""
        result = subprocess.run(
            [sys.executable, CLI_SCRIPT, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "主命令 --help 应返回 0"
        assert "FoundationPose" in result.stdout or "命令行" in result.stdout

    def test_pick_help(self):
        """测试 pick 子命令 --help"""
        result = subprocess.run(
            [sys.executable, CLI_SCRIPT, "pick", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "pick --help 应返回 0"
        assert "object_ids" in result.stdout or "物体" in result.stdout

    def test_gripper_help(self):
        """测试 gripper 子命令 --help"""
        result = subprocess.run(
            [sys.executable, CLI_SCRIPT, "gripper", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "gripper --help 应返回 0"
        assert "open" in result.stdout or "close" in result.stdout

    def test_resegment_help(self):
        """测试 resegment 子命令 --help"""
        result = subprocess.run(
            [sys.executable, CLI_SCRIPT, "resegment", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "resegment --help 应返回 0"
        assert "force" in result.stdout or "分割" in result.stdout

    def test_assign_help(self):
        """测试 assign 子命令 --help"""
        result = subprocess.run(
            [sys.executable, CLI_SCRIPT, "assign", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "assign --help 应返回 0"
        assert (
            "meshes" in result.stdout
            or "masks" in result.stdout
            or "模型" in result.stdout
        )

    def test_status_help(self):
        """测试 status 子命令 --help"""
        result = subprocess.run(
            [sys.executable, CLI_SCRIPT, "status", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "status --help 应返回 0"
        assert "状态" in result.stdout or "status" in result.stdout

    def test_param_help(self):
        """测试 param 子命令 --help"""
        result = subprocess.run(
            [sys.executable, CLI_SCRIPT, "param", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "param --help 应返回 0"
        assert "name" in result.stdout or "参数" in result.stdout


class TestCLIErrors:
    """测试 CLI 错误处理"""

    def test_invalid_subcommand(self):
        """测试无效子命令返回非零退出码"""
        result = subprocess.run(
            [sys.executable, CLI_SCRIPT, "invalid_command"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "无效子命令应返回非零退出码"

    def test_no_subcommand(self):
        """测试无子命令显示帮助"""
        result = subprocess.run(
            [sys.executable, CLI_SCRIPT],
            capture_output=True,
            text=True,
        )
        # 无子命令应显示帮助并返回 2 (argparse 标准行为)
        assert result.returncode == 2
        assert (
            "usage" in result.stdout.lower()
            or "usage" in result.stderr.lower()
            or "用法" in result.stdout
            or "用法" in result.stderr
        )


class TestCLISubcommands:
    """测试 CLI 子命令存在性"""

    def test_all_subcommands_exist(self):
        """测试所有 6 个子命令都存在"""
        subcommands = ["pick", "gripper", "resegment", "assign", "status", "param"]

        for cmd in subcommands:
            result = subprocess.run(
                [sys.executable, CLI_SCRIPT, cmd, "--help"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"子命令 '{cmd}' 不存在或 --help 失败"
