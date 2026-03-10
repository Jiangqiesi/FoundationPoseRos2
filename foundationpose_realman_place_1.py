#!/usr/bin/env python3
"""
FoundationPose MoveIt2 Controller for RealMan Robot (Real Hardware)
基于仿真版本修改，使用 RealMan SDK 控制真实机械臂。
"""
"""
使用方法示例：
1. 抓取工件1 2并放置到底座5位置
python foundationpose_moveit2_controller_place_realman.py \
  --robot-ip 192.168.0.17 \
  --objects 1 2 5

2. 依次抓取并放置所有工件
python foundationpose_moveit2_controller_place_realman.py --robot-ip 192.168.0.17 --objects 1 2 3 4 5
"""

import math
import sys
import argparse
import time
import threading
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from scipy.spatial.transform import Rotation as R

# RealMan SDK
try:
    from realman.RealMan import RM_controller
    from Robotic_Arm.rm_robot_interface import rm_thread_mode_e
except Exception as exc:
    print(f"导入 RealMan SDK 失败: {exc}", file=sys.stderr)
    print("请确保已安装 RealMan SDK", file=sys.stderr)
    sys.exit(1)


def _deg2rad_list(vals: List[float]) -> List[float]:
    """角度转弧度"""
    return [v * math.pi / 180.0 for v in vals]


def _rad2deg_list(vals: List[float]) -> List[float]:
    """弧度转角度"""
    return [v * 180.0 / math.pi for v in vals]


def _format_list(vals: List[float], precision: int = 2) -> str:
    return "[" + ", ".join(f"{v:.{precision}f}" for v in vals) + "]"


class FoundationPoseMoveIt2ControllerRealMan(Node):
    """
    Controller for robot manipulation using FoundationPose and MoveIt2 with RealMan hardware.
    """

    def __init__(
        self,
        robot_ip: str,
        object_ids: List[int],
        offset_z: float,
        approach_distance: float,
        lift_height: float,
        gripper_open_pos: float,
        gripper_close_pos: float,
        auto_move: bool = False,
        enable_grasp: bool = True,
        joint_state_hz: float = 10.0,
    ):
        super().__init__('foundationpose_moveit2_controller_realman')

        # 抓取偏移配置：目标位置相对检测位置的偏移量 (offset_x, offset_y, offset_z)
        self.grasp_offset_config = {
            1: (0.0358, 0.1116, 0.1240),
            2: (0.0858, 0.1116, 0.1240),
            3: (0.0608, 0.0606, 0.1240),
            4: (0.0608, 0.0096, 0.1240),
            5: (0.0608, 0.0606, 0.1125),
        }

        # Store parameters
        self.object_ids = object_ids
        self.auto_move = auto_move
        self.offset_z = offset_z
        self.enable_grasp = enable_grasp
        self.lift_height = lift_height
        self.approach_distance = approach_distance
        self.gripper_open_pos = gripper_open_pos
        self.gripper_close_pos = gripper_close_pos

        # State tracking
        self.latest_poses = {}
        self.subscribers = {}
        self.arm = None
        self.currently_executing = False
        self._last_valid_joints_rad: Optional[List[float]] = None
        self._last_joint_state_warn_ts = 0.0
        self._moveit_ready = False
        self._rm_api_lock = threading.Lock()

        # Callback group for concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # 在 MoveIt 初始化前先使用固定关节名发布 /joint_states，避免等待初始状态超时
        self.group_name = "rm_robot_arm"
        self.eef_link = "Link7"
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._bootstrap_pub_stop = threading.Event()
        self._bootstrap_pub_thread: Optional[threading.Thread] = None

        # ========== 连接 RealMan 机械臂 ==========
        self.get_logger().info(f"连接 RealMan 机械臂: {robot_ip}")
        try:
            self.rm_controller = RM_controller(robot_ip, rm_thread_mode_e.RM_TRIPLE_MODE_E)
            self.get_logger().info(f"连接成功，当前关节: {self.rm_controller.get_state()}")
        except Exception as e:
            self.get_logger().error(f"连接机械臂失败: {e}")
            raise

        # 在 MoveItPy 创建期间持续发布 /joint_states，确保 planning_scene_monitor 能收到实时时间戳
        self._bootstrap_pub_thread = threading.Thread(
            target=self._bootstrap_publish_joint_states,
            daemon=True,
        )
        self._bootstrap_pub_thread.start()

        # ========== 初始化 MoveIt2 ==========
        try:
            self.get_logger().info('初始化 MoveIt2...')

            # 使用与仿真版相同的 MoveIt 配置
            moveit_config = (
                MoveItConfigsBuilder(
                    robot_name="rm_robot",
                    package_name="rm_moveit2"
                )
                .robot_description(
                    file_path="config/rm_75_6f_description.urdf.xacro"
                )
                .trajectory_execution(
                    file_path="config/moveit_controllers.yaml"
                )
                .moveit_cpp(
                    file_path="config/motion_planning_python_api_tutorial.yaml"
                )
                .to_moveit_configs()
            )

            self.moveit = MoveItPy(
                node_name="moveit_py_node",
                config_dict=moveit_config.to_dict()
            )

            self.get_logger().info('MoveIt2 初始化成功')
            self.arm = self.moveit.get_planning_component(self.group_name)
            self.robot_model = self.moveit.get_robot_model()

            # JointModelGroup
            self.jmg = self.robot_model.get_joint_model_group(self.group_name)
            self.joint_names = self.jmg.active_joint_model_names

            if not self.joint_names:
                raise RuntimeError(f"无法从 JointModelGroup({self.group_name}) 获取关节名")

            self.get_logger().info(f'规划组: {self.group_name}')
            self.get_logger().info(f'关节名: {self.joint_names}')
            self._moveit_ready = True

        except Exception as e:
            self.get_logger().error(f'MoveIt2 初始化失败: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise
        finally:
            self._bootstrap_pub_stop.set()
            if self._bootstrap_pub_thread is not None:
                self._bootstrap_pub_thread.join(timeout=1.0)

        # ========== 关节状态发布（给 MoveIt 起始状态） ==========
        self.create_timer(1.0 / max(joint_state_hz, 0.1), self._publish_joint_state)

        # ========== 订阅物体位姿话题 ==========
        for obj_id in self.object_ids:
            topic_name = f'/Current_OBJ_position_{obj_id}'
            self.subscribers[obj_id] = self.create_subscription(
                PoseStamped,
                topic_name,
                lambda msg, id=obj_id: self.pose_callback(msg, id),
                10,
                callback_group=self.callback_group
            )
            self.get_logger().info(f'订阅物体 {obj_id} 位姿话题: {topic_name}')

        # Display timer
        self.timer = self.create_timer(2.0, self.display_poses)

        self.get_logger().info('控制器初始化完成')
        self.get_logger().info(f'自动移动: {"启用" if self.auto_move else "禁用"}')
        self.get_logger().info(f'抓取功能: {"启用" if self.enable_grasp else "禁用"}')

    # ========== 硬件关节状态相关 ==========
    def _looks_like_degree(self, joints: List[float]) -> bool:
        """检查角度值是否疑似为度"""
        return any(abs(v) > 10.0 for v in joints)

    def _warn_joint_state_issue(self, msg: str, throttle_sec: float = 1.0):
        """节流输出关节读取相关告警，避免日志刷屏。"""
        now = time.monotonic()
        if now - self._last_joint_state_warn_ts >= throttle_sec:
            self._last_joint_state_warn_ts = now
            self.get_logger().warn(msg)

    def _movej_with_retry(self, positions_deg: List[float], v: int = 20, retries: int = 2) -> int:
        """发送 movej，针对通信错误(-2)做短重试。"""
        last_ret = -1
        for attempt in range(retries + 1):
            with self._rm_api_lock:
                ret = self.rm_controller.movej(
                    positions_deg,
                    v=v,
                    r=0,
                    connect=0,
                    block=1,
                )
            if ret == 0:
                return 0
            last_ret = ret
            if ret == -2 and attempt < retries:
                self._warn_joint_state_issue(f"movej 通信错误(-2)，重试 {attempt + 1}/{retries}")
                time.sleep(0.05 * (attempt + 1))
                continue
            return ret
        return last_ret

    def _read_hardware_joints_rad_once(self, expected_size: Optional[int] = None) -> Optional[List[float]]:
        """读取一次硬件关节并转换为弧度；失败时返回 None。"""
        try:
            if (
                hasattr(self.rm_controller, "arm_controller")
                and hasattr(self.rm_controller.arm_controller, "rm_get_current_arm_state")
            ):
                with self._rm_api_lock:
                    code, arm_state = self.rm_controller.arm_controller.rm_get_current_arm_state()
                if code != 0:
                    self._warn_joint_state_issue(f"读取关节失败，错误码: {code}")
                    return None
                if not isinstance(arm_state, dict) or "joint" not in arm_state:
                    self._warn_joint_state_issue("读取关节失败，返回结构缺少 joint 字段")
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

        # SDK 通信失败时偶发全0状态；若上次有效状态非全0，则忽略该异常采样
        if (
            self._last_valid_joints_rad is not None
            and all(abs(v) < 1e-8 for v in joints_rad)
            and any(abs(v) > 1e-3 for v in self._last_valid_joints_rad)
        ):
            self._warn_joint_state_issue("检测到疑似异常全0关节状态，沿用上次有效状态")
            return None

        return joints_rad

    def _get_hardware_joints_rad(self, retries: int = 2, expected_size: Optional[int] = None) -> List[float]:
        """获取硬件关节值（自动检测度/弧度），失败时回退到最近有效值。"""
        for attempt in range(retries + 1):
            joints_rad = self._read_hardware_joints_rad_once(expected_size=expected_size)
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
        """仅发布关节状态到 /joint_states。"""
        try:
            joints_rad = self._get_hardware_joints_rad(expected_size=len(self.joint_names))
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
        """MoveIt 初始化前后台发布关节状态，避免 initial_state 超时。"""
        while not self._bootstrap_pub_stop.is_set():
            self._publish_joint_state_only()
            time.sleep(0.05)

    def _publish_joint_state(self, force: bool = False) -> bool:
        """发布关节状态到 /joint_states 并同步 MoveIt 起始状态。"""
        if self.currently_executing and not force:
            return False

        if not self._publish_joint_state_only():
            return False

        # 同步 MoveIt start_state
        if not self._moveit_ready or self.arm is None:
            return True
        try:
            robot_state = RobotState(self.robot_model)
            joints_rad = self._get_hardware_joints_rad(expected_size=len(self.joint_names))
            size = min(len(joints_rad), len(self.joint_names))
            robot_state.set_joint_group_active_positions(self.group_name, np.asarray(joints_rad[:size], dtype=float))
            robot_state.update()
            ok = self.arm.set_start_state(robot_state=robot_state)
            if not ok:
                self.get_logger().warn("set_start_state 返回 False")
        except Exception as exc:
            self.get_logger().warn(f"同步 start_state 失败: {exc}")
            return False
        return True

    def get_current_joint_positions(self) -> dict:
        """获取当前关节位置"""
        try:
            joints_rad = self._get_hardware_joints_rad()
        except Exception as exc:
            self._warn_joint_state_issue(f"获取当前关节失败: {exc}")
            return {}
        return dict(zip(self.joint_names, joints_rad))

    # ========== 夹爪控制（通过 RealMan SDK） ==========
    def get_gripper_position(self) -> float:
        """获取夹爪当前位置 (0.0-1.0)"""
        try:
            with self._rm_api_lock:
                return self.rm_controller.get_gripper()
        except Exception as e:
            self.get_logger().error(f'获取夹爪状态错误: {e}')
            return -1.0

    def open_gripper(self, wait=True) -> bool:
        """打开夹爪 (增量式，正值张开)"""
        self.get_logger().info('打开夹爪...')
        try:
            with self._rm_api_lock:
                self.rm_controller.set_gripper(1.0)  # 正值张开
            if wait:
                time.sleep(0.5)
            self.get_logger().info('夹爪已打开')
            return True
        except Exception as e:
            self.get_logger().error(f'打开夹爪失败: {e}')
            return False

    def close_gripper(self, wait=True) -> bool:
        """关闭夹爪 (增量式，负值关闭)"""
        self.get_logger().info('关闭夹爪...')
        try:
            with self._rm_api_lock:
                self.rm_controller.set_gripper(-1.0)  # 负值关闭
            if wait:
                time.sleep(0.5)
            self.get_logger().info('夹爪已关闭')
            return True
        except Exception as e:
            self.get_logger().error(f'关闭夹爪失败: {e}')
            return False

    # ========== 位姿回调 ==========
    def pose_callback(self, msg, object_id):
        """接收物体位姿回调"""
        self.latest_poses[object_id] = msg

        if self.auto_move and not self.currently_executing:
            self.move_to_object(object_id)

    # ========== 位姿工具函数 ==========
    def create_pose_stamped(self, x, y, z, quat_xyzw):
        """创建 PoseStamped 消息"""
        pose = PoseStamped()
        pose.header.frame_id = "base_link"  # 真机通常用 base_link
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(quat_xyzw[0])
        pose.pose.orientation.y = float(quat_xyzw[1])
        pose.pose.orientation.z = float(quat_xyzw[2])
        pose.pose.orientation.w = float(quat_xyzw[3])
        return pose

    def get_grasp_quat_from_object(self, object_orientation):
        """根据物体朝向计算夹爪姿态"""
        obj_rot = R.from_quat([
            object_orientation.x, object_orientation.y,
            object_orientation.z, object_orientation.w
        ])
        offset_rot = R.from_euler('zy', [90, 180], degrees=True)
        grasp_rot = obj_rot * offset_rot
        return grasp_rot.as_quat().tolist()

    def transform_offset_to_world(self, local_offset, orientation):
        """将物体坐标系下的偏移量转换到世界坐标系"""
        rot = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        world_offset = rot.apply(np.array(local_offset))
        return world_offset

    # ========== MoveIt 规划 ==========
    def plan_to_pose(self, target_pose_stamped):
        """使用 MoveIt 规划到目标位姿"""
        try:
            if not self._publish_joint_state(force=True):
                self.get_logger().error('无法同步有效起始关节状态，取消本次规划')
                return None
            self.arm.set_goal_state(pose_stamped_msg=target_pose_stamped, pose_link=self.eef_link)
            plan_result = self.arm.plan()

            if plan_result:
                self.get_logger().info('MoveIt 规划成功')
                return plan_result
            else:
                self.get_logger().warn('MoveIt 规划失败')
                return None

        except Exception as e:
            self.get_logger().error(f'MoveIt 规划错误: {e}')
            return None

    # ========== 轨迹执行（通过 RealMan SDK） ==========
    def execute_plan(self, plan) -> bool:
        """使用 RealMan SDK 执行规划轨迹"""
        if plan is None or plan.trajectory is None:
            self.get_logger().error('空规划或无轨迹')
            return False

        try:
            rt = plan.trajectory
            msg: RobotTrajectoryMsg = rt.get_robot_trajectory_msg()
            jt = msg.joint_trajectory
            name_to_index = {name: i for i, name in enumerate(jt.joint_names)}
            joint_order = self.joint_names or jt.joint_names
            num_points = len(jt.points)
            max_points = 10
            points_to_execute = jt.points

            if num_points > max_points:
                indices = [
                    int(i * (num_points - 1) / (max_points - 1))
                    for i in range(max_points - 1)
                ]
                indices.append(num_points - 1)
                points_to_execute = [jt.points[idx] for idx in indices]
                self.get_logger().info(
                    f'轨迹路点从 {num_points} 个均匀采样至 {len(points_to_execute)} 个'
                )
            else:
                self.get_logger().info(f'执行轨迹，共 {num_points} 个路点')

            for i, point in enumerate(points_to_execute):
                positions = [point.positions[name_to_index[n]] for n in joint_order]
                positions_deg = _rad2deg_list(positions)

                self.get_logger().info(
                    f"发送路点 {i+1}/{len(points_to_execute)}: 规划角度={_format_list(positions_deg)}"
                )

                ret = self._movej_with_retry(positions_deg, v=20, retries=2)
                if ret != 0:
                    self.get_logger().error(f"路点 {i} 执行失败，错误码: {ret}")
                    return False

            self.get_logger().info('轨迹执行完成')
            return True

        except Exception as e:
            self.get_logger().error(f'轨迹执行错误: {e}')
            return False

    def move_with_moveit(self, target_pose_stamped) -> bool:
        """规划并执行到目标位姿"""
        self.get_logger().info('使用 MoveIt 规划运动...')
        plan = self.plan_to_pose(target_pose_stamped)
        if plan:
            return self.execute_plan(plan)
        else:
            self.get_logger().warn('MoveIt 规划失败')
            return False

    # ========== 抓取序列 ==========
    def simple_grasp_sequence(self, object_id) -> bool:
        """执行简单的垂直靠近抓取序列"""
        if object_id not in self.latest_poses:
            self.get_logger().warn(f'没有物体 {object_id} 的位姿')
            return False

        self.currently_executing = True

        try:
            pose_msg = self.latest_poses[object_id]
            pos = pose_msg.pose.position
            orient = pose_msg.pose.orientation

            # 获取抓取偏移量
            local_offset = self.grasp_offset_config.get(object_id, (0.0, 0.0, 0.0))
            world_offset = self.transform_offset_to_world(local_offset, orient)

            grasp_x = pos.x + world_offset[0] + 0.027
            grasp_y = pos.y + world_offset[1]
            grasp_z = pos.z + world_offset[2] + self.offset_z
            approach_z = grasp_z + self.approach_distance
            lift_z = grasp_z + self.lift_height

            grasp_quat = self.get_grasp_quat_from_object(orient)

            self.get_logger().info(f'开始抓取物体 {object_id}')
            self.get_logger().info(f'  检测位置: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})')
            self.get_logger().info(f'  抓取位置: ({grasp_x:.3f}, {grasp_y:.3f}, {grasp_z:.3f})')

            # Step 1: 打开夹爪
            self.get_logger().info('Step 1: 打开夹爪')
            if not self.open_gripper():
                self.get_logger().error('打开夹爪失败')
                return False
            time.sleep(0.5)

            # Step 2: 移动到靠近位置
            self.get_logger().info(f'Step 2: 移动到靠近位置 (z={approach_z:.3f}m)')
            approach_pose = self.create_pose_stamped(grasp_x, grasp_y, approach_z, grasp_quat)
            if not self.move_with_moveit(approach_pose):
                self.get_logger().error('到达靠近位置失败')
                return False
            time.sleep(0.5)

            # Step 3: 下降到抓取位置
            self.get_logger().info(f'Step 3: 下降到抓取位置 (z={grasp_z:.3f}m)')
            grasp_pose = self.create_pose_stamped(grasp_x, grasp_y, grasp_z, grasp_quat)
            if not self.move_with_moveit(grasp_pose):
                self.get_logger().error('到达抓取位置失败')
                self.move_with_moveit(approach_pose)
                return False
            time.sleep(1.0)

            # Step 4: 关闭夹爪
            self.get_logger().info('Step 4: 关闭夹爪')
            if not self.close_gripper():
                self.get_logger().error('关闭夹爪失败')
                return False
            time.sleep(0.5)

            # Step 5: 提起物体
            self.get_logger().info(f'Step 5: 提起物体 (z={lift_z:.3f}m)')
            lift_pose = self.create_pose_stamped(grasp_x, grasp_y, lift_z, grasp_quat)
            if not self.move_with_moveit(lift_pose):
                self.get_logger().error('提起物体失败')
                self.open_gripper()
                return False

            self.get_logger().info(f'抓取物体 {object_id} 完成')
            return True

        finally:
            self.currently_executing = False

    def move_to_object(self, object_id) -> bool:
        """移动到物体位置，可选抓取"""
        if object_id not in self.latest_poses:
            self.get_logger().warn(f'没有物体 {object_id} 的位姿')
            return False

        try:
            if self.enable_grasp:
                return self.simple_grasp_sequence(object_id)
            else:
                pose_msg = self.latest_poses[object_id]
                pos = pose_msg.pose.position
                orient = pose_msg.pose.orientation

                local_offset = self.grasp_offset_config.get(object_id, (0.0, 0.0, 0.0))
                world_offset = self.transform_offset_to_world(local_offset, orient)

                target_x = pos.x + world_offset[0]
                target_y = pos.y + world_offset[1]
                target_z = pos.z + world_offset[2] + self.offset_z + self.approach_distance
                grasp_quat = self.get_grasp_quat_from_object(orient)

                target_pose = self.create_pose_stamped(target_x, target_y, target_z, grasp_quat)
                return self.move_with_moveit(target_pose)

        except Exception as e:
            self.get_logger().error(f'移动到物体错误: {e}')
            return False

    def move_to_target(self, source_obj_id, target_obj_id) -> bool:
        """将抓取的物体放置到目标位置"""
        self.currently_executing = True

        try:
            if target_obj_id not in self.latest_poses:
                self.get_logger().warn(f'没有目标物体 {target_obj_id} 的位姿')
                return False

            target_pose_msg = self.latest_poses[target_obj_id]
            target_pos = target_pose_msg.pose.position
            target_orient = target_pose_msg.pose.orientation

            grasp_quat = self.get_grasp_quat_from_object(target_orient)

            local_offset = self.grasp_offset_config.get(source_obj_id, (0.0, 0.0, 0.0))
            world_offset = self.transform_offset_to_world(local_offset, target_orient)

            place_x = target_pos.x + world_offset[0] + 0.027
            place_y = target_pos.y + world_offset[1]
            place_z = target_pos.z + world_offset[2] + self.offset_z + 0.005  # 微调放置高度

            self.get_logger().info(f'开始放置到目标 {target_obj_id}')
            self.get_logger().info(f'放置位置: ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})')

            # 1. 移动到上方
            approach_z = place_z + self.approach_distance
            approach_pose = self.create_pose_stamped(place_x, place_y, approach_z, grasp_quat)
            if not self.move_with_moveit(approach_pose):
                self.get_logger().error('到达放置上方位置失败')
                return False
            time.sleep(0.5)

            # 2. 下降到放置位置
            place_pose = self.create_pose_stamped(place_x, place_y, place_z, grasp_quat)
            if not self.move_with_moveit(place_pose):
                self.get_logger().error('到达放置位置失败')
                self.move_with_moveit(approach_pose)
                return False
            time.sleep(1.0)

            # 3. 打开夹爪释放物体
            self.get_logger().info('打开夹爪释放物体')
            if not self.open_gripper():
                self.get_logger().error('打开夹爪失败')
                return False
            time.sleep(0.5)

            # 4. 提起手臂
            lift_pose = self.create_pose_stamped(place_x, place_y, approach_z, grasp_quat)
            if not self.move_with_moveit(lift_pose):
                self.get_logger().error('提起手臂失败')
                return False
            time.sleep(0.5)

            self.get_logger().info(f'放置到目标 {target_obj_id} 完成')
            return True

        except Exception as e:
            self.get_logger().error(f'放置物体错误: {e}')
            return False
        finally:
            self.currently_executing = False

    def move_to_home(self) -> bool:
        """移动到初始位置"""
        self.get_logger().info('移动到 HOME 位置...')
        # 可以用预设的关节角度
        home_joints_deg = [0, 0, 0, 0, 0, 0, 0]  # 根据实际调整
        ret = self._movej_with_retry(home_joints_deg, v=20, retries=2)
        return ret == 0

    def display_poses(self):
        """显示当前物体位姿"""
        pass  # 可选：取消注释以显示调试信息

    # ========== 手动控制循环 ==========
    def manual_control_loop(self):
        """交互式手动控制"""
        print("\n手动控制模式:")
        print("- 输入多个ID (如 '1 2 3'): 抓取1和2放到3上")
        print("- 输入单个ID (如 '1'): 只抓取1")
        print("- 'status' 查看状态")
        print("- 'gripper open' / 'gripper close' 控制夹爪")
        print("- 'poses' 显示检测到的位姿")
        print("- 'joints' 显示当前关节")
        print("- 'home' 回到初始位置")
        print("- 'q' 退出")

        while rclpy.ok():
            try:
                user_input = input("\n输入命令 > ").strip()

                if user_input.lower() in ['q', 'quit']:
                    print("退出程序")
                    break

                elif user_input.lower() == 'status':
                    ordered = [i for i in self.object_ids if i in self.latest_poses] # 按输入顺序显示已检测物体
                    print(f"跟踪的物体: {ordered}")
                    continue

                elif user_input.lower() == 'gripper open':
                    if self.open_gripper():
                        print("夹爪已打开")
                    else:
                        print("打开夹爪失败")
                    continue

                elif user_input.lower() == 'gripper close':
                    if self.close_gripper():
                        print("夹爪已关闭")
                    else:
                        print("关闭夹爪失败")
                    continue

                elif user_input.lower() == 'poses':
                    if not self.latest_poses:
                        print("未检测到物体位姿")
                    else:
                        for obj_id in sorted(self.latest_poses.keys()):
                            msg = self.latest_poses[obj_id]
                            pos = msg.pose.position
                            orient = msg.pose.orientation

                            local_offset = self.grasp_offset_config.get(obj_id, (0.0, 0.0, 0.0))
                            world_offset = self.transform_offset_to_world(local_offset, orient)

                            center_x = pos.x + world_offset[0]
                            center_y = pos.y + world_offset[1]
                            center_z = pos.z + world_offset[2]
                            grasp_quat = self.get_grasp_quat_from_object(orient)

                            print(f"物体 {obj_id}:")
                            print(f"检测位置: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")
                            print(f"检测朝向: ({orient.x:.4f}, {orient.y:.4f}, {orient.z:.4f}, {orient.w:.4f})")
                            print(f"抓取中心: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")
                            print(f"抓取朝向: ({grasp_quat[0]:.4f}, {grasp_quat[1]:.4f}, {grasp_quat[2]:.4f}, {grasp_quat[3]:.4f})")
                    continue

                elif user_input.lower() == 'joints':
                    positions = self.get_current_joint_positions()
                    if not positions:
                        print("无关节数据")
                    else:
                        print("当前关节位置:")
                        for name, pos in positions.items():
                            print(f"  {name}: {pos:.4f} rad ({math.degrees(pos):.2f} deg)")
                    continue

                elif user_input.lower() == 'home':
                    if self.move_to_home():
                        print("已回到 HOME 位置")
                    else:
                        print("回到 HOME 位置失败")
                    continue

                try:
                    input_ids = [int(x) for x in user_input.split()]

                    if not input_ids:
                        continue

                    if len(input_ids) == 1:
                        obj_id = input_ids[0]
                        print(f"单物体模式: 抓取物体 {obj_id}")
                        if self.move_to_object(obj_id):
                            print(f"抓取 {obj_id} 完成")
                        else:
                            print(f"抓取 {obj_id} 失败")

                    else:
                        target_id = input_ids[-1]
                        source_ids = input_ids[:-1]

                        print(f"序列模式: 将 {source_ids} 移动到目标 {target_id}")

                        if target_id not in self.latest_poses:
                            print(f"错误: 目标物体 {target_id} 位姿未检测到!")
                            continue

                        for i, obj_id in enumerate(source_ids):
                            print(f"\n--- 序列 {i+1}/{len(source_ids)}: 处理物体 {obj_id} ---")

                            if obj_id not in self.latest_poses:
                                print(f"跳过物体 {obj_id}: 位姿未检测到")
                                continue

                            print(f"步骤 1: 抓取物体 {obj_id}...")
                            if not self.move_to_object(obj_id):
                                print(f"抓取物体 {obj_id} 失败，跳过")
                                self.open_gripper()
                                continue

                            print(f"步骤 2: 放置到目标 {target_id}...")
                            if not self.move_to_target(obj_id, target_id):
                                print(f"放置物体 {obj_id} 失败")
                                self.open_gripper()
                            else:
                                print(f"成功将 {obj_id} 放到 {target_id}")

                        print("\n所有任务完成")

                except ValueError:
                    print("无效输入，请输入物体ID或命令")

            except KeyboardInterrupt:
                print("\n收到退出信号")
                break
            except Exception as e:
                print(f"错误: {e}")

    def cleanup(self):
        """清理资源"""
        self._bootstrap_pub_stop.set()
        self.get_logger().info("清理资源...")


def main():
    parser = argparse.ArgumentParser(description='FoundationPose MoveIt2 Controller for RealMan Robot')
    parser.add_argument('--robot-ip', default='192.168.0.17',
                       help='RealMan 机械臂 IP 地址 (默认: 192.168.0.17)')
    parser.add_argument('--objects', nargs='+', type=int, default=[1],
                       help='订阅的物体 ID (默认: [1])')
    parser.add_argument('--auto-move', action='store_true',
                       help='启用自动移动模式')
    parser.add_argument('--offset-z', type=float, default=0.13,
                       help='抓取 Z 轴偏移 (默认: 0.15m)')
    parser.add_argument('--approach-distance', type=float, default=0.1,
                       help='物体上方靠近距离 (默认: 0.1m)')
    parser.add_argument('--lift-height', type=float, default=0.1,
                       help='抓取后提升高度 (默认: 0.1m)')
    parser.add_argument('--disable-grasp', action='store_true',
                       help='禁用抓取（只移动到位置）')
    parser.add_argument('--gripper-open-pos', type=float, default=0.8,
                       help='夹爪打开位置 (默认: 0.8)')
    parser.add_argument('--gripper-close-pos', type=float, default=0.28,
                       help='夹爪关闭位置 (默认: 0.28)')
    parser.add_argument('--joint-state-hz', type=float, default=10.0,
                       help='/joint_states 发布频率 (默认: 10.0)')

    args = parser.parse_args()

    rclpy.init()

    controller = None
    try:
        controller = FoundationPoseMoveIt2ControllerRealMan(
            robot_ip=args.robot_ip,
            object_ids=args.objects,
            auto_move=args.auto_move,
            offset_z=args.offset_z,
            enable_grasp=not args.disable_grasp,
            lift_height=args.lift_height,
            approach_distance=args.approach_distance,
            gripper_open_pos=args.gripper_open_pos,
            gripper_close_pos=args.gripper_close_pos,
            joint_state_hz=args.joint_state_hz,
        )

        print(f"监听物体 {args.objects} 位姿...")
        print("按 Ctrl+C 退出")

        if args.auto_move:
            rclpy.spin(controller)
        else:
            ros_thread = threading.Thread(target=lambda: rclpy.spin(controller))
            ros_thread.daemon = True
            ros_thread.start()

            controller.manual_control_loop()

    except KeyboardInterrupt:
        print("\n收到退出信号，关闭中...")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller:
            controller.cleanup()
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
