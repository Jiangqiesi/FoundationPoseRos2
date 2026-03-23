#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoundationPose ROS2 客户端 API
封装所有 Action/Service 调用为 Python API
"""

import os
import sys
import time
import threading
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from typing import List, Dict, Optional, Callable, Any
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 动态添加 foundationpose_msgs 到路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_msgs_install_path = os.path.join(
    _script_dir,
    "foundationpose_msgs",
    "install",
    "foundationpose_msgs",
    "lib",
    "python3.10",
    "site-packages",
)
if os.path.exists(_msgs_install_path):
    sys.path.insert(0, _msgs_install_path)

from foundationpose_msgs.action import PickPlace
from foundationpose_msgs.srv import (
    Resegment,
    AssignModels,
    GripperControl,
    UpdateParams,
    PerceptionStatus,
    ControllerStatus,
)


class FoundationPoseClient:
    """
    FoundationPose ROS2 客户端

    封装所有 ROS2 Action 和 Service 调用，提供简洁的 Python API。
    支持上下文管理器协议，自动处理资源清理。

    示例:
        with FoundationPoseClient() as client:
            if client.wait_for_servers(timeout_sec=10.0):
                result = client.pick_place(object_ids=[1, 2])
                print(result)
    """

    def __init__(
        self, node_name: str = "foundationpose_client", config_path: str = None
    ):
        # 初始化 rclpy（如果尚未初始化）
        # 禁用 rclpy 内置 SIGINT handler，避免 Ctrl+C 时 rclpy 抢先 shutdown
        # 导致 node context 失效，cancel_current_action() 无法正常工作
        if not rclpy.ok():
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

        self._node = Node(node_name)
        self._config = self._load_config(config_path)
        self._bridge = CvBridge()
        self._session_id: int = 0

        # 从配置读取服务名
        server_config = self._config.get("server", {})
        controller_config = server_config.get("controller", {})
        perception_config = server_config.get("perception_services", {})

        # Action Client
        action_name = controller_config.get("action_name", "/robot/pick_place")
        self._pick_place_action_client = ActionClient(
            self._node, PickPlace, action_name
        )

        # Service Clients
        self._gripper_client = self._node.create_client(
            GripperControl,
            controller_config.get("gripper_service", "/controller/gripper"),
        )

        self._update_params_client = self._node.create_client(
            UpdateParams,
            controller_config.get("params_service", "/controller/update_params"),
        )

        self._controller_status_client = self._node.create_client(
            ControllerStatus,
            controller_config.get("status_service", "/controller/get_status"),
        )

        self._resegment_client = self._node.create_client(
            Resegment, perception_config.get("resegment", "/perception/resegment")
        )

        self._assign_models_client = self._node.create_client(
            AssignModels,
            perception_config.get("assign_models", "/perception/assign_models"),
        )

        self._perception_status_client = self._node.create_client(
            PerceptionStatus, perception_config.get("status", "/perception/get_status")
        )

        self._current_goal_handle = None

        # 图像订阅：mask 可视化 / mask 标签 / 位姿可视化
        self._latest_masks_vis: Optional[np.ndarray] = None
        self._latest_masks_label: Optional[np.ndarray] = None
        self._latest_pose_vis: Optional[np.ndarray] = None
        self._image_lock = threading.Lock()

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._masks_vis_sub = self._node.create_subscription(
            Image,
            "/perception/masks_visualization",
            self._on_masks_vis,
            latched_qos,
        )
        self._masks_label_sub = self._node.create_subscription(
            Image,
            "/perception/masks_label",
            self._on_masks_label,
            latched_qos,
        )
        self._pose_vis_sub = self._node.create_subscription(
            Image,
            "/perception/pose_visualization",
            self._on_pose_vis,
            10,
        )

        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin_background, daemon=True)
        self._spin_thread.start()

    def _spin_background(self) -> None:
        try:
            self._executor.spin()
        except Exception:
            pass

    def _wait_for_future(self, future: Future, timeout_sec: float = 10.0) -> bool:
        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > timeout_sec:
                return False
            time.sleep(0.01)
        return True

    def _on_masks_vis(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_masks_vis = self._bridge.imgmsg_to_cv2(msg, "rgb8")

    def _on_masks_label(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_masks_label = self._bridge.imgmsg_to_cv2(msg, "mono8")

    def _on_pose_vis(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_pose_vis = self._bridge.imgmsg_to_cv2(msg, "rgb8")

    @property
    def session_id(self) -> int:
        return self._session_id

    def get_masks_visualization(self) -> Optional[np.ndarray]:
        with self._image_lock:
            return (
                self._latest_masks_vis.copy()
                if self._latest_masks_vis is not None
                else None
            )

    def get_masks_label(self) -> Optional[np.ndarray]:
        with self._image_lock:
            return (
                self._latest_masks_label.copy()
                if self._latest_masks_label is not None
                else None
            )

    def get_pose_visualization(self) -> Optional[np.ndarray]:
        with self._image_lock:
            return (
                self._latest_pose_vis.copy()
                if self._latest_pose_vis is not None
                else None
            )

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path is None:
            # 尝试从多个位置查找配置文件
            possible_paths = [
                "config/pick_place.yaml",
                "../FoundationPoseROS2/config/pick_place.yaml",
                os.path.join(
                    os.path.dirname(_script_dir),
                    "FoundationPoseROS2",
                    "config",
                    "pick_place.yaml",
                ),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                self._node.get_logger().warn(f"配置文件加载失败: {e}")

        return {}

    def wait_for_servers(self, timeout_sec: float = 30.0) -> bool:
        """
        等待所有服务端就绪

        Args:
            timeout_sec: 超时时间（秒）

        Returns:
            所有服务端是否就绪

        Raises:
            RuntimeError: 超时后服务端仍未就绪
        """
        self._node.get_logger().info("等待服务端就绪...")

        clients = [
            ("PickPlace Action", self._pick_place_action_client),
            ("Gripper Service", self._gripper_client),
            ("UpdateParams Service", self._update_params_client),
            ("ControllerStatus Service", self._controller_status_client),
            ("Resegment Service", self._resegment_client),
            ("AssignModels Service", self._assign_models_client),
            ("PerceptionStatus Service", self._perception_status_client),
        ]

        all_ready = True
        for name, client in clients:
            if hasattr(client, "wait_for_server"):
                # Action client
                ready = client.wait_for_server(timeout_sec=timeout_sec)
            else:
                # Service client
                ready = client.wait_for_service(timeout_sec=timeout_sec)

            if ready:
                self._node.get_logger().info(f"✓ {name} 就绪")
            else:
                self._node.get_logger().error(f"✗ {name} 超时")
                all_ready = False

        if not all_ready:
            raise RuntimeError(f"服务端未在 {timeout_sec} 秒内就绪")

        return True

    def get_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            包含感知和控制器状态的字典
        """
        result = {}

        # 查询感知状态
        perception_req = PerceptionStatus.Request()
        perception_req.dummy = 0
        perception_future = self._perception_status_client.call_async(perception_req)
        self._wait_for_future(perception_future, timeout_sec=5.0)

        if perception_future.result() is not None:
            resp = perception_future.result()
            result["perception"] = {
                "ready": resp.ready,
                "segmentation_done": resp.segmentation_done,
                "num_tracked_objects": resp.num_tracked_objects,
                "tracked_object_ids": list(resp.tracked_object_ids),
                "tracked_mesh_names": list(resp.tracked_mesh_names),
            }
        else:
            result["perception"] = {"error": "感知状态查询失败"}

        # 查询控制器状态
        controller_req = ControllerStatus.Request()
        controller_req.dummy = 0
        controller_future = self._controller_status_client.call_async(controller_req)
        self._wait_for_future(controller_future, timeout_sec=5.0)

        if controller_future.result() is not None:
            resp = controller_future.result()
            result["controller"] = {
                "robot_connected": resp.robot_connected,
                "moveit_ready": resp.moveit_ready,
                "currently_executing": resp.currently_executing,
                "current_joints": list(resp.current_joints),
                "joint_names": list(resp.joint_names),
            }
        else:
            result["controller"] = {"error": "控制器状态查询失败"}

        return result

    def resegment(self, force: bool = False) -> Dict[str, Any]:
        request = Resegment.Request()
        request.force = force

        future = self._resegment_client.call_async(request)
        self._wait_for_future(future, timeout_sec=10.0)

        if future.result() is not None:
            resp = future.result()
            self._session_id = int(resp.session_id)
            return {
                "success": resp.success,
                "message": resp.message,
                "session_id": resp.session_id,
                "num_masks": resp.num_masks,
            }
        else:
            raise TimeoutError("重新分割请求超时")

    def assign_models(
        self,
        mesh_paths: List[str],
        mask_indices: List[int],
        session_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if len(mesh_paths) != len(mask_indices):
            raise ValueError("mesh_paths 和 mask_indices 长度必须相同")

        request = AssignModels.Request()
        request.mesh_paths = mesh_paths
        request.mask_indices = mask_indices
        request.session_id = session_id if session_id is not None else self._session_id

        future = self._assign_models_client.call_async(request)
        self._wait_for_future(future, timeout_sec=10.0)

        if future.result() is not None:
            resp = future.result()
            return {"success": resp.success, "message": resp.message}
        else:
            raise TimeoutError("模型分配请求超时")

    def pick_place(
        self,
        object_ids: List[int],
        target_id: int = 0,
        enable_grasp: bool = True,
        offset_z: float = 0.133,
        approach_distance: float = 0.1,
        lift_height: float = 0.1,
        feedback_callback: Optional[Callable[[Any], None]] = None,
    ) -> Dict[str, Any]:
        """
        同步执行抓取放置任务

        Args:
            object_ids: 要抓取的物体 ID 列表
            target_id: 放置目标 ID
            enable_grasp: 是否启用夹爪控制
            offset_z: Z 轴放置偏移
            approach_distance: 靠近物体时的距离
            lift_height: 提升高度
            feedback_callback: 反馈回调函数

        Returns:
            包含 success, message, completed_objects, failed_objects 的字典
        """
        goal = PickPlace.Goal()
        goal.object_ids = object_ids
        goal.target_id = target_id
        goal.enable_grasp = enable_grasp
        goal.offset_z = offset_z
        goal.approach_distance = approach_distance
        goal.lift_height = lift_height

        self._node.get_logger().info(f"发送抓取放置目标: 物体 {object_ids}")

        send_goal_future = self._pick_place_action_client.send_goal_async(
            goal, feedback_callback=feedback_callback
        )
        self._wait_for_future(send_goal_future, timeout_sec=30.0)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            return {
                "success": False,
                "message": "目标被服务端拒绝",
                "completed_objects": [],
                "failed_objects": object_ids,
            }

        self._current_goal_handle = goal_handle
        self._node.get_logger().info("目标已接受，等待结果...")

        result_future = goal_handle.get_result_async()
        self._wait_for_future(result_future, timeout_sec=300.0)

        result = result_future.result().result
        self._current_goal_handle = None

        return {
            "success": result.success,
            "message": result.message,
            "completed_objects": list(result.completed_objects),
            "failed_objects": list(result.failed_objects),
        }

    def pick_place_async(
        self,
        object_ids: List[int],
        target_id: int = 0,
        enable_grasp: bool = True,
        offset_z: float = 0.133,
        approach_distance: float = 0.1,
        lift_height: float = 0.1,
        feedback_callback: Optional[Callable[[Any], None]] = None,
    ) -> Future:
        """
        异步执行抓取放置任务

        Args:
            object_ids: 要抓取的物体 ID 列表
            target_id: 放置目标 ID
            enable_grasp: 是否启用夹爪控制
            offset_z: Z 轴放置偏移
            approach_distance: 靠近物体时的距离
            lift_height: 提升高度
            feedback_callback: 反馈回调函数

        Returns:
            Future 对象，可用于异步获取结果
        """
        goal = PickPlace.Goal()
        goal.object_ids = object_ids
        goal.target_id = target_id
        goal.enable_grasp = enable_grasp
        goal.offset_z = offset_z
        goal.approach_distance = approach_distance
        goal.lift_height = lift_height

        return self._pick_place_action_client.send_goal_async(
            goal, feedback_callback=feedback_callback
        )

    def cancel_current_action(self) -> bool:
        """
        取消当前正在执行的 Action

        Returns:
            是否成功发送取消请求
        """
        if self._current_goal_handle is None:
            self._node.get_logger().warn("没有正在执行的 Action")
            return False

        self._node.get_logger().info("取消当前 Action...")
        cancel_future = self._current_goal_handle.cancel_goal_async()
        self._wait_for_future(cancel_future, timeout_sec=5.0)

        if cancel_future.result() is not None:
            self._node.get_logger().info("取消请求已发送")
            return True
        else:
            self._node.get_logger().error("取消请求超时")
            return False

    def open_gripper(self) -> Dict[str, Any]:
        """
        打开夹爪

        Returns:
            包含 success 和 message 的字典
        """
        request = GripperControl.Request()
        request.action = "open"
        request.value = 1.0

        future = self._gripper_client.call_async(request)
        self._wait_for_future(future, timeout_sec=5.0)

        if future.result() is not None:
            resp = future.result()
            return {"success": resp.success, "message": resp.message}
        else:
            raise TimeoutError("夹爪控制请求超时")

    def close_gripper(self) -> Dict[str, Any]:
        """
        关闭夹爪

        Returns:
            包含 success 和 message 的字典
        """
        request = GripperControl.Request()
        request.action = "close"
        request.value = 1.0

        future = self._gripper_client.call_async(request)
        self._wait_for_future(future, timeout_sec=5.0)

        if future.result() is not None:
            resp = future.result()
            return {"success": resp.success, "message": resp.message}
        else:
            raise TimeoutError("夹爪控制请求超时")

    def update_param(self, param_name: str, value: float) -> Dict[str, Any]:
        """
        更新运动参数

        Args:
            param_name: 参数名称
            value: 参数值

        Returns:
            包含 success 和 message 的字典
        """
        request = UpdateParams.Request()
        request.param_name = param_name
        request.value = str(value)

        future = self._update_params_client.call_async(request)
        self._wait_for_future(future, timeout_sec=5.0)

        if future.result() is not None:
            resp = future.result()
            return {"success": resp.success, "message": resp.message}
        else:
            raise TimeoutError("参数更新请求超时")

    def shutdown(self):
        try:
            if self._current_goal_handle is not None:
                self.cancel_current_action()
        except Exception:
            pass

        try:
            self._executor.shutdown()
        except Exception:
            pass

        try:
            self._node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.shutdown()
        return False


if __name__ == "__main__":
    # 简单测试
    with FoundationPoseClient() as client:
        try:
            print("等待服务端...")
            client.wait_for_servers(timeout_sec=5.0)
            print("所有服务端就绪")

            status = client.get_status()
            print(f"系统状态: {status}")
        except Exception as e:
            print(f"测试失败: {e}")
