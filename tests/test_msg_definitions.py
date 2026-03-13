#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 foundationpose_msgs 消息定义
"""

import pytest
import sys
import os


# 检查消息包是否已编译
msgs_install_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "foundationpose_msgs",
    "install",
    "foundationpose_msgs",
    "lib",
    "python3.10",
    "site-packages",
)
MSGS_AVAILABLE = os.path.exists(msgs_install_path)


@pytest.mark.skipif(not MSGS_AVAILABLE, reason="foundationpose_msgs 未编译")
class TestMessageImports:
    """测试消息类型可导入"""

    def test_import_pickplace_action(self):
        """测试导入 PickPlace Action"""
        try:
            from foundationpose_msgs.action import PickPlace

            assert PickPlace is not None
        except ImportError as e:
            pytest.fail(f"无法导入 PickPlace Action: {e}")

    def test_import_services(self):
        """测试导入所有 Service 类型"""
        try:
            from foundationpose_msgs.srv import (
                Resegment,
                AssignModels,
                GripperControl,
                UpdateParams,
                PerceptionStatus,
                ControllerStatus,
            )

            assert Resegment is not None
            assert AssignModels is not None
            assert GripperControl is not None
            assert UpdateParams is not None
            assert PerceptionStatus is not None
            assert ControllerStatus is not None
        except ImportError as e:
            pytest.fail(f"无法导入 Service 类型: {e}")


@pytest.mark.skipif(not MSGS_AVAILABLE, reason="foundationpose_msgs 未编译")
class TestPickPlaceAction:
    """测试 PickPlace Action 字段"""

    def test_pickplace_goal_fields(self):
        """测试 PickPlace.Goal 字段"""
        from foundationpose_msgs.action import PickPlace

        goal = PickPlace.Goal()

        # 检查必需字段存在
        assert hasattr(goal, "object_ids"), "Goal 缺少 object_ids 字段"
        assert hasattr(goal, "target_id"), "Goal 缺少 target_id 字段"
        assert hasattr(goal, "enable_grasp"), "Goal 缺少 enable_grasp 字段"
        assert hasattr(goal, "offset_z"), "Goal 缺少 offset_z 字段"
        assert hasattr(goal, "approach_distance"), "Goal 缺少 approach_distance 字段"
        assert hasattr(goal, "lift_height"), "Goal 缺少 lift_height 字段"

    def test_pickplace_result_fields(self):
        """测试 PickPlace.Result 字段"""
        from foundationpose_msgs.action import PickPlace

        result = PickPlace.Result()

        # 检查必需字段存在
        assert hasattr(result, "success"), "Result 缺少 success 字段"
        assert hasattr(result, "message"), "Result 缺少 message 字段"
        assert hasattr(result, "completed_objects"), (
            "Result 缺少 completed_objects 字段"
        )
        assert hasattr(result, "failed_objects"), "Result 缺少 failed_objects 字段"

    def test_pickplace_feedback_fields(self):
        """测试 PickPlace.Feedback 字段"""
        from foundationpose_msgs.action import PickPlace

        feedback = PickPlace.Feedback()

        # 检查必需字段存在
        assert hasattr(feedback, "current_object_id"), (
            "Feedback 缺少 current_object_id 字段"
        )
        assert hasattr(feedback, "stage"), "Feedback 缺少 stage 字段"
        assert hasattr(feedback, "objects_completed"), (
            "Feedback 缺少 objects_completed 字段"
        )
        assert hasattr(feedback, "objects_total"), "Feedback 缺少 objects_total 字段"
        assert hasattr(feedback, "detail"), "Feedback 缺少 detail 字段"


@pytest.mark.skipif(not MSGS_AVAILABLE, reason="foundationpose_msgs 未编译")
class TestServiceDefinitions:
    """测试 Service 定义字段"""

    def test_resegment_service(self):
        """测试 Resegment Service"""
        from foundationpose_msgs.srv import Resegment

        request = Resegment.Request()
        response = Resegment.Response()

        assert hasattr(request, "force"), "Resegment.Request 缺少 force 字段"
        assert hasattr(response, "success"), "Resegment.Response 缺少 success 字段"
        assert hasattr(response, "message"), "Resegment.Response 缺少 message 字段"

    def test_assign_models_service(self):
        """测试 AssignModels Service"""
        from foundationpose_msgs.srv import AssignModels

        request = AssignModels.Request()
        response = AssignModels.Response()

        assert hasattr(request, "mesh_paths"), (
            "AssignModels.Request 缺少 mesh_paths 字段"
        )
        assert hasattr(request, "mask_indices"), (
            "AssignModels.Request 缺少 mask_indices 字段"
        )
        assert hasattr(response, "success"), "AssignModels.Response 缺少 success 字段"
        assert hasattr(response, "message"), "AssignModels.Response 缺少 message 字段"

    def test_gripper_control_service(self):
        """测试 GripperControl Service"""
        from foundationpose_msgs.srv import GripperControl

        request = GripperControl.Request()
        response = GripperControl.Response()

        assert hasattr(request, "action"), "GripperControl.Request 缺少 action 字段"
        assert hasattr(request, "value"), "GripperControl.Request 缺少 value 字段"
        assert hasattr(response, "success"), "GripperControl.Response 缺少 success 字段"
        assert hasattr(response, "message"), "GripperControl.Response 缺少 message 字段"

    def test_perception_status_service(self):
        """测试 PerceptionStatus Service"""
        from foundationpose_msgs.srv import PerceptionStatus

        request = PerceptionStatus.Request()
        response = PerceptionStatus.Response()

        assert hasattr(request, "dummy"), "PerceptionStatus.Request 缺少 dummy 字段"
        assert hasattr(response, "ready"), "PerceptionStatus.Response 缺少 ready 字段"
        assert hasattr(response, "segmentation_done"), (
            "PerceptionStatus.Response 缺少 segmentation_done 字段"
        )
        assert hasattr(response, "num_tracked_objects"), (
            "PerceptionStatus.Response 缺少 num_tracked_objects 字段"
        )
        assert hasattr(response, "tracked_object_ids"), (
            "PerceptionStatus.Response 缺少 tracked_object_ids 字段"
        )

    def test_controller_status_service(self):
        """测试 ControllerStatus Service"""
        from foundationpose_msgs.srv import ControllerStatus

        request = ControllerStatus.Request()
        response = ControllerStatus.Response()

        assert hasattr(request, "dummy"), "ControllerStatus.Request 缺少 dummy 字段"
        assert hasattr(response, "robot_connected"), (
            "ControllerStatus.Response 缺少 robot_connected 字段"
        )
        assert hasattr(response, "moveit_ready"), (
            "ControllerStatus.Response 缺少 moveit_ready 字段"
        )
        assert hasattr(response, "currently_executing"), (
            "ControllerStatus.Response 缺少 currently_executing 字段"
        )
