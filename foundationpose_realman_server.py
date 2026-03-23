#!/usr/bin/env python3

import argparse
import importlib
import math
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
import foundationpose_msgs.action as fp_action
import foundationpose_msgs.srv as fp_srv
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState

from foundationpose_logging import setup_file_logging

PickPlace: Any = getattr(fp_action, "PickPlace")
ControllerStatus: Any = getattr(fp_srv, "ControllerStatus")
GripperControl: Any = getattr(fp_srv, "GripperControl")
PerceptionStatus: Any = getattr(fp_srv, "PerceptionStatus")
UpdateParams: Any = getattr(fp_srv, "UpdateParams")
RobotState = importlib.import_module("moveit.core.robot_state").RobotState

try:
    from realman.RealMan import RM_controller
    from Robotic_Arm.rm_robot_interface import rm_thread_mode_e
except Exception as exc:
    print(f"导入 RealMan SDK 失败: {exc}", file=sys.stderr)
    print("请确保已安装 RealMan SDK", file=sys.stderr)
    sys.exit(1)


def _deg2rad_list(vals: List[float]) -> List[float]:
    return [v * math.pi / 180.0 for v in vals]


def _rad2deg_list(vals: List[float]) -> List[float]:
    return [v * 180.0 / math.pi for v in vals]


def _format_list(vals: List[float], precision: int = 2) -> str:
    return "[" + ", ".join(f"{v:.{precision}f}" for v in vals) + "]"


def _parse_bool(text: str) -> Optional[bool]:
    lower = text.strip().lower()
    if lower in ("1", "true", "yes", "y", "on"):
        return True
    if lower in ("0", "false", "no", "n", "off"):
        return False
    return None


class FoundationPoseRealManServer(Node):
    def __init__(
        self,
        robot_ip: str,
        object_ids: List[int],
        offset_z: float,
        approach_distance: float,
        lift_height: float,
        enable_grasp: bool = True,
        joint_state_hz: float = 10.0,
    ):
        super().__init__("foundationpose_realman_server")

        self.grasp_offset_config: Dict[str, Tuple[float, float, float]] = {
            "g0801": (0.0358, 0.1116, 0.1240),
            "g0802": (0.0858, 0.1116, 0.1240),
            "g0701": (0.0608, 0.0606, 0.1240),
            "g0601": (0.0608, 0.0096, 0.1240),
            "g0101": (0.0608, 0.0606, 0.1125),
        }

        self._obj_id_to_mesh_name: Dict[int, str] = {}

    def _lookup_grasp_offset(self, mesh_name: str) -> Tuple[float, float, float]:
        for key, offset in self.grasp_offset_config.items():
            if key in mesh_name:
                return offset
        return (0.0, 0.0, 0.0)

        self.robot_ip = robot_ip
        self.object_ids = object_ids
        self.offset_z = offset_z
        self.enable_grasp = enable_grasp
        self.lift_height = lift_height
        self.approach_distance = approach_distance
        self.joint_state_hz = joint_state_hz

        self.latest_poses: Dict[int, PoseStamped] = {}
        self.subscribers: Dict[int, object] = {}
        self.arm = None
        self.robot_connected = False
        self.currently_executing = False
        self._cancel_requested = threading.Event()

        self._last_valid_joints_rad: Optional[List[float]] = None
        self._last_joint_state_warn_ts = 0.0
        self._moveit_ready = False
        self._rm_api_lock = threading.Lock()
        self._warned_deg = False
        self._connect_lock = threading.Lock()

        self.callback_group = ReentrantCallbackGroup()

        self.group_name = "rm_robot_arm"
        self.eef_link = "Link7"
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._bootstrap_pub_stop = threading.Event()
        self._bootstrap_pub_thread: Optional[threading.Thread] = None
        self.joint_state_timer: Optional[object] = None
        self.rm_controller: Any = None
        self.moveit: Any = None
        self.robot_model: Any = None
        self.jmg: Any = None

        # 不在启动时连接机械臂，等待客户端请求时再懒加载连接
        self.get_logger().info(
            f"服务端启动，机械臂 IP: {robot_ip}（延迟连接，等待客户端请求）"
        )

        for obj_id in self.object_ids:
            topic_name = f"/Current_OBJ_position_{obj_id}"
            self.subscribers[obj_id] = self.create_subscription(
                PoseStamped,
                topic_name,
                lambda msg, id=obj_id: self.pose_callback(msg, id),
                10,
                callback_group=self.callback_group,
            )
            self.get_logger().info(f"订阅物体 {obj_id} 位姿话题: {topic_name}")

        self.timer = self.create_timer(2.0, self.display_poses)

        self.pick_place_action_server = ActionServer(
            self,
            PickPlace,
            "/robot/pick_place",
            execute_callback=self.execute_pick_place,
            goal_callback=self.pick_place_goal_callback,
            cancel_callback=self.pick_place_cancel_callback,
            callback_group=self.callback_group,
        )

        self.gripper_service = self.create_service(
            GripperControl,
            "/controller/gripper",
            self.handle_gripper_service,
            callback_group=self.callback_group,
        )
        self.update_params_service = self.create_service(
            UpdateParams,
            "/controller/update_params",
            self.handle_update_params_service,
            callback_group=self.callback_group,
        )
        self.get_status_service = self.create_service(
            ControllerStatus,
            "/controller/get_status",
            self.handle_get_status_service,
            callback_group=self.callback_group,
        )

        self._perception_status_client = self.create_client(
            PerceptionStatus,
            "/perception/get_status",
            callback_group=self.callback_group,
        )

        self.get_logger().info("控制服务端初始化完成（机械臂未连接，按需连接）")
        self.get_logger().info(
            f"抓取功能默认状态: {'启用' if self.enable_grasp else '禁用'}"
        )

        setup_file_logging(self, "logs/realman")

    def _refresh_obj_id_to_mesh_name(self) -> bool:
        if not self._perception_status_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("感知状态服务不可用，无法刷新 mesh 名称映射")
            return False

        req = PerceptionStatus.Request()
        req.dummy = 0
        future = self._perception_status_client.call_async(req)

        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > 5.0:
                self.get_logger().warn("查询感知状态超时")
                return False
            time.sleep(0.01)

        resp = future.result()
        if resp is None:
            self.get_logger().warn("查询感知状态返回 None")
            return False

        mapping: Dict[int, str] = {}
        for oid, name in zip(resp.tracked_object_ids, resp.tracked_mesh_names):
            mapping[int(oid)] = str(name)
        self._obj_id_to_mesh_name = mapping
        self.get_logger().info(f"已刷新物体ID→mesh名称映射: {mapping}")
        return True

    def _ensure_robot_connected(self) -> bool:
        """懒加载连接机械臂和初始化 MoveIt2。线程安全，仅在首次需要时执行。
        返回 True 表示连接成功，False 表示连接失败。"""
        if self.robot_connected and self._moveit_ready:
            return True

        with self._connect_lock:
            # 双重检查，避免并发重复连接
            if self.robot_connected and self._moveit_ready:
                return True

            self.get_logger().info(f"正在连接 RealMan 机械臂: {self.robot_ip}")
            try:
                self.rm_controller = RM_controller(
                    self.robot_ip, rm_thread_mode_e.RM_TRIPLE_MODE_E
                )
                self.robot_connected = True
                self.get_logger().info(
                    f"连接成功，当前关节: {self.rm_controller.get_state()}"
                )
            except Exception as e:
                self.get_logger().error(f"连接机械臂失败: {e}")
                self.robot_connected = False
                return False

            self._bootstrap_pub_stop.clear()
            self._bootstrap_pub_thread = threading.Thread(
                target=self._bootstrap_publish_joint_states, daemon=True
            )
            self._bootstrap_pub_thread.start()

            try:
                self.get_logger().info("初始化 MoveIt2...")

                moveit_config = (
                    MoveItConfigsBuilder(
                        robot_name="rm_robot", package_name="rm_moveit2"
                    )
                    .robot_description(
                        file_path="config/rm_75_6f_description.urdf.xacro"
                    )
                    .trajectory_execution(file_path="config/moveit_controllers.yaml")
                    .moveit_cpp(
                        file_path="config/motion_planning_python_api_tutorial.yaml"
                    )
                    .to_moveit_configs()
                )

                self.moveit = MoveItPy(
                    node_name="moveit_py_node", config_dict=moveit_config.to_dict()
                )
                self.get_logger().info("MoveIt2 初始化成功")
                self.arm = self.moveit.get_planning_component(self.group_name)
                self.robot_model = self.moveit.get_robot_model()
                self.jmg = self.robot_model.get_joint_model_group(self.group_name)
                self.joint_names = self.jmg.active_joint_model_names

                if not self.joint_names:
                    raise RuntimeError(
                        f"无法从 JointModelGroup({self.group_name}) 获取关节名"
                    )

                self.get_logger().info(f"规划组: {self.group_name}")
                self.get_logger().info(f"关节名: {self.joint_names}")
                self._moveit_ready = True

            except Exception as e:
                self.get_logger().error(f"MoveIt2 初始化失败: {e}")
                import traceback

                self.get_logger().error(traceback.format_exc())
                # 机械臂已连接但 MoveIt 失败，断开连接回退到未连接状态
                self.robot_connected = False
                self._moveit_ready = False
                self.rm_controller = None
                return False
            finally:
                self._bootstrap_pub_stop.set()
                if self._bootstrap_pub_thread is not None:
                    self._bootstrap_pub_thread.join(timeout=1.0)

            self._set_joint_state_timer(self.joint_state_hz)
            self.get_logger().info("机械臂连接和 MoveIt2 初始化完成")
            return True

    def _looks_like_degree(self, joints: List[float]) -> bool:
        return any(abs(v) > 10.0 for v in joints)

    def _warn_joint_state_issue(self, msg: str, throttle_sec: float = 1.0):
        now = time.monotonic()
        if now - self._last_joint_state_warn_ts >= throttle_sec:
            self._last_joint_state_warn_ts = now
            self.get_logger().warn(msg)

    def _movej_with_retry(
        self, positions_deg: List[float], v: int = 20, retries: int = 2
    ) -> int:
        last_ret = -1
        for attempt in range(retries + 1):
            with self._rm_api_lock:
                ret = self.rm_controller.movej(
                    positions_deg, v=v, r=0, connect=0, block=1
                )
            if ret == 0:
                return 0
            last_ret = ret
            if ret == -2 and attempt < retries:
                self._warn_joint_state_issue(
                    f"movej 通信错误(-2)，重试 {attempt + 1}/{retries}"
                )
                time.sleep(0.05 * (attempt + 1))
                continue
            return ret
        return last_ret

    def _read_hardware_joints_rad_once(
        self, expected_size: Optional[int] = None
    ) -> Optional[List[float]]:
        try:
            if hasattr(self.rm_controller, "arm_controller") and hasattr(
                self.rm_controller.arm_controller, "rm_get_current_arm_state"
            ):
                with self._rm_api_lock:
                    code, arm_state = (
                        self.rm_controller.arm_controller.rm_get_current_arm_state()
                    )
                if code != 0:
                    self._warn_joint_state_issue(f"读取关节失败，错误码: {code}")
                    return None
                if not isinstance(arm_state, dict) or "joint" not in arm_state:
                    self._warn_joint_state_issue(
                        "读取关节失败，返回结构缺少 joint 字段"
                    )
                    return None
                raw = [float(v) for v in arm_state["joint"]]
            else:
                with self._rm_api_lock:
                    raw = [float(v) for v in self.rm_controller.get_state()]
        except Exception as exc:
            self._warn_joint_state_issue(f"读取关节异常: {exc}")
            return None

        if expected_size is None:
            expected_size = len(self.joint_names)
        if len(raw) < expected_size:
            self._warn_joint_state_issue(
                f"读取关节数量不足: got={len(raw)}, expect={expected_size}"
            )
            return None
        raw = raw[:expected_size]

        if not all(math.isfinite(v) for v in raw):
            self._warn_joint_state_issue("读取关节包含非有限值，已忽略")
            return None

        if self._looks_like_degree(raw):
            if not hasattr(self, "_warned_deg"):
                self._warned_deg = True
                self.get_logger().info("检测到关节值疑似度，自动转弧度")
            joints_rad = _deg2rad_list(raw)
        else:
            joints_rad = raw

        if (
            self._last_valid_joints_rad is not None
            and all(abs(v) < 1e-8 for v in joints_rad)
            and any(abs(v) > 1e-3 for v in self._last_valid_joints_rad)
        ):
            self._warn_joint_state_issue("检测到疑似异常全0关节状态，沿用上次有效状态")
            return None

        return joints_rad

    def _get_hardware_joints_rad(
        self, retries: int = 2, expected_size: Optional[int] = None
    ) -> List[float]:
        for attempt in range(retries + 1):
            joints_rad = self._read_hardware_joints_rad_once(
                expected_size=expected_size
            )
            if joints_rad is not None:
                self._last_valid_joints_rad = joints_rad
                return joints_rad
            if attempt < retries:
                time.sleep(0.02)

        if self._last_valid_joints_rad is not None:
            self._warn_joint_state_issue("读取关节失败，回退到上次有效关节状态")
            return list(self._last_valid_joints_rad)

        raise RuntimeError("无法获取有效的机械臂关节状态")

    def _publish_joint_state_only(self) -> bool:
        if not self.robot_connected or self.rm_controller is None:
            return False

        try:
            joints_rad = self._get_hardware_joints_rad(
                expected_size=len(self.joint_names)
            )
        except Exception as exc:
            self._warn_joint_state_issue(f"发布关节状态失败: {exc}")
            return False

        size = min(len(joints_rad), len(self.joint_names))
        if size <= 0:
            self._warn_joint_state_issue("发布关节状态失败：关节数组为空")
            return False

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names[:size]
        msg.position = joints_rad[:size]
        self.joint_state_pub.publish(msg)
        return True

    def _bootstrap_publish_joint_states(self):
        while not self._bootstrap_pub_stop.is_set():
            self._publish_joint_state_only()
            time.sleep(0.05)

    def _publish_joint_state(self, force: bool = False) -> bool:
        if self.currently_executing and not force:
            return False

        if not self._publish_joint_state_only():
            return False

        if not self._moveit_ready or self.arm is None:
            return True
        try:
            robot_state = RobotState(self.robot_model)
            joints_rad = self._get_hardware_joints_rad(
                expected_size=len(self.joint_names)
            )
            size = min(len(joints_rad), len(self.joint_names))
            robot_state.set_joint_group_active_positions(
                self.group_name, np.asarray(joints_rad[:size], dtype=float)
            )
            robot_state.update()
            ok = self.arm.set_start_state(robot_state=robot_state)
            if not ok:
                self.get_logger().warn("set_start_state 返回 False")
        except Exception as exc:
            self.get_logger().warn(f"同步 start_state 失败: {exc}")
            return False
        return True

    def get_current_joint_positions(self) -> Dict[str, float]:
        try:
            joints_rad = self._get_hardware_joints_rad()
        except Exception as exc:
            self._warn_joint_state_issue(f"获取当前关节失败: {exc}")
            return {}
        return dict(zip(self.joint_names, joints_rad))

    def _set_joint_state_timer(self, joint_state_hz: float):
        self.joint_state_hz = float(joint_state_hz)
        if self.joint_state_timer is not None:
            cancel_fn = getattr(self.joint_state_timer, "cancel", None)
            if callable(cancel_fn):
                cancel_fn()
            self.joint_state_timer = None

        if self.joint_state_hz > 0.0:
            period = 1.0 / max(self.joint_state_hz, 0.1)
            self.joint_state_timer = self.create_timer(
                period, self._publish_joint_state
            )
            self.get_logger().info(
                f"/joint_states 定时同步已启用: {self.joint_state_hz:.2f} Hz"
            )
        else:
            self.get_logger().info("/joint_states 定时同步已禁用，仅在规划前同步")

    def get_gripper_position(self) -> float:
        try:
            with self._rm_api_lock:
                return self.rm_controller.get_gripper()
        except Exception as e:
            self.get_logger().error(f"获取夹爪状态错误: {e}")
            return -1.0

    def open_gripper(self, wait: bool = True) -> bool:
        self.get_logger().info("打开夹爪...")
        try:
            with self._rm_api_lock:
                self.rm_controller.set_gripper(1.0)
            if wait:
                time.sleep(0.5)
            self.get_logger().info("夹爪已打开")
            return True
        except Exception as e:
            self.get_logger().error(f"打开夹爪失败: {e}")
            return False

    def close_gripper(self, wait: bool = True) -> bool:
        self.get_logger().info("关闭夹爪...")
        try:
            with self._rm_api_lock:
                self.rm_controller.set_gripper(-1.0)
            if wait:
                time.sleep(0.5)
            self.get_logger().info("夹爪已关闭")
            return True
        except Exception as e:
            self.get_logger().error(f"关闭夹爪失败: {e}")
            return False

    def pose_callback(self, msg, object_id: int):
        self.latest_poses[object_id] = msg

    def create_pose_stamped(
        self, x: float, y: float, z: float, quat_xyzw: List[float]
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(quat_xyzw[0])
        pose.pose.orientation.y = float(quat_xyzw[1])
        pose.pose.orientation.z = float(quat_xyzw[2])
        pose.pose.orientation.w = float(quat_xyzw[3])
        return pose

    def get_grasp_quat_from_object(self, object_orientation) -> List[float]:
        obj_rot = R.from_quat(
            [
                object_orientation.x,
                object_orientation.y,
                object_orientation.z,
                object_orientation.w,
            ]
        )
        offset_rot = R.from_euler("zy", [90, 180], degrees=True)
        grasp_rot = obj_rot * offset_rot
        return grasp_rot.as_quat().tolist()

    def transform_offset_to_world(
        self, local_offset: Tuple[float, float, float], orientation
    ) -> List[float]:
        rot = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        world_offset = rot.apply(np.array(local_offset))
        return world_offset.tolist()

    def plan_to_pose(self, target_pose_stamped: PoseStamped):
        try:
            if self.arm is None:
                self.get_logger().error("MoveIt 规划组件未初始化")
                return None
            if not self._publish_joint_state(force=True):
                self.get_logger().error("无法同步有效起始关节状态，取消本次规划")
                return None
            self.arm.set_goal_state(
                pose_stamped_msg=target_pose_stamped, pose_link=self.eef_link
            )
            plan_result = self.arm.plan()

            if plan_result:
                self.get_logger().info("MoveIt 规划成功")
                return plan_result

            self.get_logger().warn("MoveIt 规划失败")
            return None

        except Exception as e:
            self.get_logger().error(f"MoveIt 规划错误: {e}")
            return None

    def execute_plan(self, plan) -> bool:
        if plan is None or plan.trajectory is None:
            self.get_logger().error("空规划或无轨迹")
            return False

        try:
            rt = plan.trajectory
            msg: RobotTrajectoryMsg = rt.get_robot_trajectory_msg()
            jt = msg.joint_trajectory
            name_to_index = {name: i for i, name in enumerate(jt.joint_names)}
            joint_order = (
                list(self.joint_names) if self.joint_names else list(jt.joint_names)
            )
            num_points = len(jt.points)
            max_points = 10
            points_all = list(jt.points)
            points_to_execute = points_all

            if num_points > max_points:
                indices = [
                    int(i * (num_points - 1) / (max_points - 1))
                    for i in range(max_points - 1)
                ]
                indices.append(num_points - 1)
                points_to_execute = [points_all[idx] for idx in indices]
                self.get_logger().info(
                    f"轨迹路点从 {num_points} 个均匀采样至 {len(points_to_execute)} 个"
                )
            else:
                self.get_logger().info(f"执行轨迹，共 {num_points} 个路点")

            for i, point in enumerate(points_to_execute):
                point_positions = list(getattr(point, "positions", []))
                positions = [point_positions[name_to_index[n]] for n in joint_order]
                positions_deg = _rad2deg_list(positions)
                self.get_logger().info(
                    f"发送路点 {i + 1}/{len(points_to_execute)}: 规划角度={_format_list(positions_deg)}"
                )

                ret = self._movej_with_retry(positions_deg, v=20, retries=2)
                if ret != 0:
                    self.get_logger().error(f"路点 {i} 执行失败，错误码: {ret}")
                    return False

            self.get_logger().info("轨迹执行完成")
            return True

        except Exception as e:
            self.get_logger().error(f"轨迹执行错误: {e}")
            return False

    def move_with_moveit(self, target_pose_stamped: PoseStamped) -> bool:
        self.get_logger().info("使用 MoveIt 规划运动...")
        plan = self.plan_to_pose(target_pose_stamped)
        if plan:
            return self.execute_plan(plan)
        self.get_logger().warn("MoveIt 规划失败")
        return False

    def _publish_pick_place_feedback(
        self,
        goal_handle,
        stage: str,
        current_object_id: int,
        objects_completed: int,
        objects_total: int,
        detail: str,
    ):
        feedback = PickPlace.Feedback()
        feedback.stage = stage
        feedback.current_object_id = int(current_object_id)
        feedback.objects_completed = int(objects_completed)
        feedback.objects_total = int(objects_total)
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    def _should_cancel(self, goal_handle, detail: str = "收到取消请求") -> bool:
        canceled = self._cancel_requested.is_set() or goal_handle.is_cancel_requested
        if canceled:
            self.get_logger().warn(detail)
            try:
                self.open_gripper(wait=False)
            except Exception:
                pass
        return canceled

    def simple_grasp_sequence(
        self,
        object_id: int,
        goal_handle,
        completed_count: int,
        total_count: int,
    ) -> bool:
        if object_id not in self.latest_poses:
            self.get_logger().warn(f"没有物体 {object_id} 的位姿")
            return False

        pose_msg = self.latest_poses[object_id]
        pos = pose_msg.pose.position
        orient = pose_msg.pose.orientation

        mesh_name = self._obj_id_to_mesh_name.get(object_id, "")
        local_offset = self._lookup_grasp_offset(mesh_name)
        world_offset = self.transform_offset_to_world(local_offset, orient)

        grasp_x = pos.x + world_offset[0] + 0.028
        grasp_y = pos.y + world_offset[1] + 0.006
        grasp_z = pos.z + world_offset[2] + self.offset_z
        approach_z = grasp_z + self.approach_distance
        lift_z = grasp_z + self.lift_height
        grasp_quat = self.get_grasp_quat_from_object(orient)

        self.get_logger().info(f"开始抓取物体 {object_id}")
        self.get_logger().info(
            f"  mesh名称: '{mesh_name}', 本地偏移: {local_offset}, 世界偏移: [{world_offset[0]:.4f}, {world_offset[1]:.4f}, {world_offset[2]:.4f}]"
        )
        self.get_logger().info(
            f"  抓取四元数(xyzw): [{grasp_quat[0]:.4f}, {grasp_quat[1]:.4f}, {grasp_quat[2]:.4f}, {grasp_quat[3]:.4f}]"
        )
        self.get_logger().info(f"  识别位置: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        self.get_logger().info(
            f"  抓取位置: ({grasp_x:.3f}, {grasp_y:.3f}, {grasp_z:.3f})"
        )
        self.get_logger().info(f"  靠近高度: {approach_z:.3f}, 提起高度: {lift_z:.3f}")

        if self._should_cancel(goal_handle, "抓取前检测到取消"):
            return False

        self._publish_pick_place_feedback(
            goal_handle,
            "opening_gripper",
            object_id,
            completed_count,
            total_count,
            "打开夹爪",
        )
        if not self.open_gripper():
            self.get_logger().error("打开夹爪失败")
            return False

        if self._should_cancel(goal_handle, "打开夹爪后检测到取消"):
            return False

        self._publish_pick_place_feedback(
            goal_handle,
            "approaching",
            object_id,
            completed_count,
            total_count,
            f"移动到靠近位置 z={approach_z:.3f}",
        )
        approach_pose = self.create_pose_stamped(
            grasp_x, grasp_y, approach_z, grasp_quat
        )
        if not self.move_with_moveit(approach_pose):
            self.get_logger().error("到达靠近位置失败")
            return False

        if self._should_cancel(goal_handle, "靠近后检测到取消"):
            return False

        self._publish_pick_place_feedback(
            goal_handle,
            "descending",
            object_id,
            completed_count,
            total_count,
            f"下降到抓取位置 z={grasp_z:.3f}",
        )
        grasp_pose = self.create_pose_stamped(grasp_x, grasp_y, grasp_z, grasp_quat)
        if not self.move_with_moveit(grasp_pose):
            self.get_logger().error("到达抓取位置失败")
            self.move_with_moveit(approach_pose)
            return False

        if self._should_cancel(goal_handle, "下降后检测到取消"):
            return False

        self._publish_pick_place_feedback(
            goal_handle,
            "closing_gripper",
            object_id,
            completed_count,
            total_count,
            "关闭夹爪",
        )
        if not self.close_gripper():
            self.get_logger().error("关闭夹爪失败")
            return False

        if self._should_cancel(goal_handle, "闭合夹爪后检测到取消"):
            return False

        self._publish_pick_place_feedback(
            goal_handle,
            "lifting",
            object_id,
            completed_count,
            total_count,
            f"提起物体 z={lift_z:.3f}",
        )
        lift_pose = self.create_pose_stamped(grasp_x, grasp_y, lift_z, grasp_quat)
        if not self.move_with_moveit(lift_pose):
            self.get_logger().error("提起物体失败")
            self.open_gripper()
            return False

        self.get_logger().info(f"抓取物体 {object_id} 完成")
        return True

    def move_to_target(
        self,
        source_obj_id: int,
        target_obj_id: int,
        goal_handle,
        completed_count: int,
        total_count: int,
    ) -> bool:
        if target_obj_id not in self.latest_poses:
            self.get_logger().warn(f"没有目标物体 {target_obj_id} 的位姿")
            return False

        target_pose_msg = self.latest_poses[target_obj_id]
        target_pos = target_pose_msg.pose.position
        target_orient = target_pose_msg.pose.orientation
        grasp_quat = self.get_grasp_quat_from_object(target_orient)

        mesh_name = self._obj_id_to_mesh_name.get(source_obj_id, "")
        local_offset = self._lookup_grasp_offset(mesh_name)
        world_offset = self.transform_offset_to_world(local_offset, target_orient)

        place_x = target_pos.x + world_offset[0] + 0.028
        place_y = target_pos.y + world_offset[1] + 0.006
        place_z = target_pos.z + world_offset[2] + self.offset_z + 0.003
        approach_z = place_z + self.approach_distance

        self.get_logger().info(f"开始放置到目标 {target_obj_id}")
        self.get_logger().info(
            f"  源物体: {source_obj_id}, mesh名称: '{mesh_name}', 本地偏移: {local_offset}, 世界偏移: [{world_offset[0]:.4f}, {world_offset[1]:.4f}, {world_offset[2]:.4f}]"
        )
        self.get_logger().info(
            f"  目标识别位置: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})"
        )
        self.get_logger().info(
            f"  放置位置: ({place_x:.3f}, {place_y:.3f}, {place_z:.3f}), 靠近高度: {approach_z:.3f}"
        )

        if self._should_cancel(goal_handle, "放置前检测到取消"):
            return False

        self._publish_pick_place_feedback(
            goal_handle,
            "moving_to_place",
            source_obj_id,
            completed_count,
            total_count,
            f"移动到放置上方 z={approach_z:.3f}",
        )
        approach_pose = self.create_pose_stamped(
            place_x, place_y, approach_z, grasp_quat
        )
        if not self.move_with_moveit(approach_pose):
            self.get_logger().error("到达放置上方位置失败")
            return False

        if self._should_cancel(goal_handle, "放置靠近后检测到取消"):
            return False

        self._publish_pick_place_feedback(
            goal_handle,
            "placing",
            source_obj_id,
            completed_count,
            total_count,
            f"下降并释放到 z={place_z:.3f}",
        )
        place_pose = self.create_pose_stamped(place_x, place_y, place_z, grasp_quat)
        if not self.move_with_moveit(place_pose):
            self.get_logger().error("到达放置位置失败")
            self.move_with_moveit(approach_pose)
            return False

        if not self.open_gripper():
            self.get_logger().error("打开夹爪失败")
            return False

        lift_pose = self.create_pose_stamped(place_x, place_y, approach_z, grasp_quat)
        if not self.move_with_moveit(lift_pose):
            self.get_logger().error("提起手臂失败")
            return False

        self.get_logger().info(f"放置到目标 {target_obj_id} 完成")
        return True

    def pick_place_goal_callback(self, goal_request) -> GoalResponse:
        if self.currently_executing:
            self.get_logger().warn("当前已有任务执行中，拒绝新 Goal")
            return GoalResponse.REJECT

        if not goal_request.object_ids:
            self.get_logger().warn("收到空 object_ids，拒绝 Goal")
            return GoalResponse.REJECT

        self.get_logger().info(
            f"接收 PickPlace Goal: objects={list(goal_request.object_ids)}, "
            f"target={goal_request.target_id}, enable_grasp={goal_request.enable_grasp}"
        )
        return GoalResponse.ACCEPT

    def pick_place_cancel_callback(self, _goal_handle) -> CancelResponse:
        self.get_logger().warn("收到取消请求，准备安全停止")
        self._cancel_requested.set()
        if self.robot_connected:
            self.open_gripper(wait=False)
        return CancelResponse.ACCEPT

    def execute_pick_place(self, goal_handle):
        result = PickPlace.Result()
        completed_objects: List[int] = []
        failed_objects: List[int] = []

        if not self._ensure_robot_connected():
            result.success = False
            result.message = "机械臂连接失败，无法执行 PickPlace"
            result.completed_objects = []
            result.failed_objects = [int(i) for i in goal_handle.request.object_ids]
            goal_handle.abort()
            return result

        self._refresh_obj_id_to_mesh_name()

        default_enable_grasp = self.enable_grasp
        default_offset_z = self.offset_z
        default_approach_distance = self.approach_distance
        default_lift_height = self.lift_height

        self.currently_executing = True
        self._cancel_requested.clear()

        try:
            self.enable_grasp = bool(goal_handle.request.enable_grasp)
            self.offset_z = float(goal_handle.request.offset_z)
            self.approach_distance = float(goal_handle.request.approach_distance)
            self.lift_height = float(goal_handle.request.lift_height)

            object_ids = [int(i) for i in goal_handle.request.object_ids]
            target_id = int(goal_handle.request.target_id)
            total_count = len(object_ids)

            self.get_logger().info(
                f"开始执行 PickPlace: objects={object_ids}, target_id={target_id}, total={total_count}"
            )
            self.get_logger().info(
                f"  参数: enable_grasp={self.enable_grasp}, offset_z={self.offset_z:.4f}, "
                f"approach_distance={self.approach_distance:.4f}, lift_height={self.lift_height:.4f}"
            )

            for idx, object_id in enumerate(object_ids):
                if self._should_cancel(goal_handle, f"执行物体 {object_id} 前取消"):
                    result.success = False
                    result.message = "任务已取消"
                    result.completed_objects = completed_objects
                    result.failed_objects = failed_objects + object_ids[idx:]
                    goal_handle.canceled()
                    return result

                if object_id not in self.latest_poses:
                    self.get_logger().warn(f"物体 {object_id} 位姿不可用，跳过")
                    failed_objects.append(object_id)
                    continue

                if self.enable_grasp:
                    grasp_ok = self.simple_grasp_sequence(
                        object_id=object_id,
                        goal_handle=goal_handle,
                        completed_count=len(completed_objects),
                        total_count=total_count,
                    )
                else:
                    grasp_ok = True

                if not grasp_ok:
                    failed_objects.append(object_id)
                    self.open_gripper(wait=False)
                    continue

                if target_id > 0:
                    place_ok = self.move_to_target(
                        source_obj_id=object_id,
                        target_obj_id=target_id,
                        goal_handle=goal_handle,
                        completed_count=len(completed_objects),
                        total_count=total_count,
                    )
                    if not place_ok:
                        failed_objects.append(object_id)
                        self.open_gripper(wait=False)
                        continue

                completed_objects.append(object_id)
                self._publish_pick_place_feedback(
                    goal_handle,
                    "placing" if target_id > 0 else "lifting",
                    object_id,
                    len(completed_objects),
                    total_count,
                    f"物体 {object_id} 已完成",
                )

            success = len(completed_objects) == total_count and len(failed_objects) == 0
            result.success = success
            result.completed_objects = completed_objects
            result.failed_objects = failed_objects

            if self._should_cancel(goal_handle, "任务尾阶段检测到取消"):
                result.success = False
                result.message = "任务已取消"
                goal_handle.canceled()
                return result

            if success:
                result.message = "全部任务执行成功"
                goal_handle.succeed()
            else:
                result.message = (
                    f"部分任务失败: 完成 {len(completed_objects)}/{total_count}, "
                    f"失败对象 {failed_objects}"
                )
                goal_handle.abort()

            return result

        except Exception as e:
            self.get_logger().error(f"执行 PickPlace 异常: {e}")
            result.success = False
            result.message = f"执行异常: {e}"
            result.completed_objects = completed_objects
            result.failed_objects = failed_objects
            goal_handle.abort()
            return result

        finally:
            self.currently_executing = False
            self._cancel_requested.clear()
            self.enable_grasp = default_enable_grasp
            self.offset_z = default_offset_z
            self.approach_distance = default_approach_distance
            self.lift_height = default_lift_height

    def handle_gripper_service(self, request, response):
        if not self._ensure_robot_connected():
            response.success = False
            response.message = "机械臂未连接，无法控制夹爪"
            return response

        action = request.action.strip().lower()
        if action == "open":
            ok = self.open_gripper()
            response.success = ok
            response.message = "夹爪已打开" if ok else "打开夹爪失败"
            return response

        if action == "close":
            ok = self.close_gripper()
            response.success = ok
            response.message = "夹爪已关闭" if ok else "关闭夹爪失败"
            return response

        response.success = False
        response.message = f"未知 action: {request.action}，仅支持 open/close"
        return response

    def handle_update_params_service(self, request, response):
        param_name = request.param_name.strip()
        value = request.value.strip()

        try:
            if param_name == "offset_z":
                self.offset_z = float(value)
                response.success = True
                response.message = f"offset_z 已更新为 {self.offset_z:.4f}"
                return response

            if param_name == "approach_distance":
                self.approach_distance = float(value)
                response.success = True
                response.message = (
                    f"approach_distance 已更新为 {self.approach_distance:.4f}"
                )
                return response

            if param_name == "lift_height":
                self.lift_height = float(value)
                response.success = True
                response.message = f"lift_height 已更新为 {self.lift_height:.4f}"
                return response

            if param_name == "enable_grasp":
                parsed = _parse_bool(value)
                if parsed is None:
                    response.success = False
                    response.message = "enable_grasp 仅支持 true/false/1/0"
                    return response
                self.enable_grasp = parsed
                response.success = True
                response.message = f"enable_grasp 已更新为 {self.enable_grasp}"
                return response

            if param_name == "joint_state_hz":
                hz = float(value)
                self._set_joint_state_timer(hz)
                response.success = True
                response.message = f"joint_state_hz 已更新为 {self.joint_state_hz:.2f}"
                return response

            response.success = False
            response.message = (
                "不支持的参数，支持: offset_z, approach_distance, "
                "lift_height, enable_grasp, joint_state_hz"
            )
            return response
        except Exception as e:
            response.success = False
            response.message = f"更新参数失败: {e}"
            return response

    def handle_get_status_service(self, _request, response):
        response.robot_connected = bool(self.robot_connected)
        response.moveit_ready = bool(self._moveit_ready)
        response.currently_executing = bool(self.currently_executing)
        if self.robot_connected:
            current_joints = self.get_current_joint_positions()
            response.joint_names = list(current_joints.keys())
            response.current_joints = [float(v) for v in current_joints.values()]
        else:
            response.joint_names = []
            response.current_joints = []
        return response

    def display_poses(self):
        pass

    def cleanup(self):
        self._bootstrap_pub_stop.set()
        if self.joint_state_timer is not None:
            cancel_fn = getattr(self.joint_state_timer, "cancel", None)
            if callable(cancel_fn):
                cancel_fn()
            self.joint_state_timer = None
        self.get_logger().info("清理资源...")


def main():
    parser = argparse.ArgumentParser(description="FoundationPose RealMan 控制服务端")
    parser.add_argument(
        "--robot-ip",
        default="192.168.0.17",
        help="RealMan 机械臂 IP 地址 (默认: 192.168.0.17)",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        type=int,
        default=[1],
        help="订阅的物体 ID (默认: [1])",
    )
    parser.add_argument(
        "--offset-z",
        type=float,
        default=0.133,
        help="抓取 Z 轴偏移 (默认: 0.133m)",
    )
    parser.add_argument(
        "--approach-distance",
        type=float,
        default=0.1,
        help="物体上方靠近距离 (默认: 0.1m)",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.1,
        help="抓取后提升高度 (默认: 0.1m)",
    )
    parser.add_argument(
        "--disable-grasp",
        action="store_true",
        help="禁用抓取（仅执行路径动作）",
    )
    parser.add_argument(
        "--joint-state-hz",
        type=float,
        default=10.0,
        help="/joint_states 发布频率，<=0 表示禁用定时同步，仅规划前同步 (默认: 10.0)",
    )

    args = parser.parse_args()

    rclpy.init()
    server = None
    try:
        server = FoundationPoseRealManServer(
            robot_ip=args.robot_ip,
            object_ids=args.objects,
            offset_z=args.offset_z,
            approach_distance=args.approach_distance,
            lift_height=args.lift_height,
            enable_grasp=not args.disable_grasp,
            joint_state_hz=args.joint_state_hz,
        )

        server.get_logger().info(f"监听物体位姿: {args.objects}")
        server.get_logger().info("服务端就绪: Action=/robot/pick_place")
        server.get_logger().info(
            "Services=/controller/gripper, /controller/update_params, /controller/get_status"
        )
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(server)
        executor.spin()

    except KeyboardInterrupt:
        print("\n收到退出信号，关闭中...")
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if server is not None:
            server.cleanup()
            server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
