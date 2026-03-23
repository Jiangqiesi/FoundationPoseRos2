#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 FoundationPoseClient API
"""

import pytest
import inspect
from unittest.mock import Mock, patch, MagicMock


class TestFoundationPoseClientStructure:
    """测试客户端类结构"""

    @patch("foundationpose_client.MultiThreadedExecutor")
    @patch("foundationpose_client.rclpy")
    @patch("foundationpose_client.Node")
    @patch("foundationpose_client.ActionClient")
    def test_client_has_all_public_methods(
        self, mock_action, mock_node, mock_rclpy, mock_executor
    ):
        """测试客户端有所有 10 个公开方法"""
        mock_rclpy.ok.return_value = False
        mock_node_instance = MagicMock()
        mock_node.return_value = mock_node_instance

        from foundationpose_client import FoundationPoseClient

        client = FoundationPoseClient()

        # 10 个公开方法
        expected_methods = [
            "wait_for_servers",
            "get_status",
            "resegment",
            "assign_models",
            "pick_place",
            "pick_place_async",
            "cancel_current_action",
            "open_gripper",
            "close_gripper",
            "update_param",
        ]

        for method_name in expected_methods:
            assert hasattr(client, method_name), f"缺少方法: {method_name}"
            assert callable(getattr(client, method_name)), f"{method_name} 不可调用"

    @patch("foundationpose_client.MultiThreadedExecutor")
    @patch("foundationpose_client.rclpy")
    @patch("foundationpose_client.Node")
    @patch("foundationpose_client.ActionClient")
    def test_client_is_context_manager(
        self, mock_action, mock_node, mock_rclpy, mock_executor
    ):
        """测试客户端支持上下文管理器协议"""
        mock_rclpy.ok.return_value = False
        mock_node_instance = MagicMock()
        mock_node.return_value = mock_node_instance

        from foundationpose_client import FoundationPoseClient

        client = FoundationPoseClient()

        assert hasattr(client, "__enter__"), "缺少 __enter__ 方法"
        assert hasattr(client, "__exit__"), "缺少 __exit__ 方法"
        assert callable(client.__enter__), "__enter__ 不可调用"
        assert callable(client.__exit__), "__exit__ 不可调用"

    @patch("foundationpose_client.MultiThreadedExecutor")
    @patch("foundationpose_client.rclpy")
    @patch("foundationpose_client.Node")
    @patch("foundationpose_client.ActionClient")
    def test_methods_have_type_hints(
        self, mock_action, mock_node, mock_rclpy, mock_executor
    ):
        """测试方法有类型提示"""
        mock_rclpy.ok.return_value = False
        mock_node_instance = MagicMock()
        mock_node.return_value = mock_node_instance

        from foundationpose_client import FoundationPoseClient

        client = FoundationPoseClient()

        # 检查关键方法的返回类型注解
        methods_with_return_types = [
            "wait_for_servers",
            "get_status",
            "resegment",
            "assign_models",
            "pick_place",
        ]

        for method_name in methods_with_return_types:
            method = getattr(client, method_name)
            sig = inspect.signature(method)
            assert sig.return_annotation != inspect.Signature.empty, (
                f"{method_name} 缺少返回类型注解"
            )

    @patch("foundationpose_client.MultiThreadedExecutor")
    @patch("foundationpose_client.rclpy")
    @patch("foundationpose_client.Node")
    @patch("foundationpose_client.ActionClient")
    def test_client_initialization(
        self, mock_action, mock_node, mock_rclpy, mock_executor
    ):
        """测试客户端初始化"""
        mock_rclpy.ok.return_value = False
        mock_node_instance = MagicMock()
        mock_node.return_value = mock_node_instance

        from foundationpose_client import FoundationPoseClient

        # 测试默认初始化
        client = FoundationPoseClient()
        assert client is not None

        # 测试自定义节点名
        client2 = FoundationPoseClient(node_name="test_client")
        assert client2 is not None

    @patch("foundationpose_client.MultiThreadedExecutor")
    @patch("foundationpose_client.rclpy")
    @patch("foundationpose_client.Node")
    @patch("foundationpose_client.ActionClient")
    def test_context_manager_usage(
        self, mock_action, mock_node, mock_rclpy, mock_executor
    ):
        """测试上下文管理器使用"""
        mock_rclpy.ok.return_value = False
        mock_node_instance = MagicMock()
        mock_node.return_value = mock_node_instance

        from foundationpose_client import FoundationPoseClient

        # 测试 with 语句
        with FoundationPoseClient() as client:
            assert client is not None
            assert hasattr(client, "_node")

    @patch("foundationpose_client.MultiThreadedExecutor")
    @patch("foundationpose_client.rclpy")
    @patch("foundationpose_client.Node")
    @patch("foundationpose_client.ActionClient")
    def test_shutdown_method(self, mock_action, mock_node, mock_rclpy, mock_executor):
        """测试 shutdown 方法"""
        mock_rclpy.ok.return_value = False
        mock_node_instance = MagicMock()
        mock_node.return_value = mock_node_instance

        from foundationpose_client import FoundationPoseClient

        client = FoundationPoseClient()

        # 测试 shutdown 不抛出异常
        try:
            client.shutdown()
        except Exception as e:
            pytest.fail(f"shutdown() 抛出异常: {e}")
