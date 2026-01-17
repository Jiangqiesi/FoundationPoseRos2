#!/usr/bin/env python3
# 基础版：订阅手动发布的 PoseStamped，调用 MoveIt 规划并用 RealMan SDK 执行
import argparse
import math
import sys
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy
from moveit.core.robot_model import RobotModel
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg

try:
    from realman.RealMan import RM_controller
    from Robotic_Arm.rm_robot_interface import rm_thread_mode_e
except Exception as exc:
    print(f"导入 RealMan SDK 失败: {exc}", file=sys.stderr)
    sys.exit(1)

# 角度转弧度
def _deg2rad_list(vals: List[float]) -> List[float]:
    return [v * math.pi / 180.0 for v in vals]


# 弧度转角度
def _rad2deg_list(vals: List[float]) -> List[float]:
    return [v * 180.0 / math.pi for v in vals]


class SimpleMoveItController(Node):
    def __init__(
        self,
        robot_ip: str,
        pose_topic: str,
        group_name: str,
        eef_link: str,
        joint_names: Optional[List[str]] = None,
        joint_state_hz: float = 10.0,
    ):
        """
        初始化 SimpleMoveItController 控制器
        
        Args:
            robot_ip (str): 机器人控制器的IP地址
            pose_topic (str): 订阅目标位姿的话题名称
            group_name (str): MoveIt中使用的规划组名称
            eef_link (str): 机器人末端执行器链接名称
            joint_names (Optional[List[str]]): 关节名称列表，默认为None时会从JointModelGroup获取
            joint_state_hz (float): 关节状态发布的频率，默认为10.0Hz
        """
        super().__init__("simple_moveit_controller")

        # 连接硬件
        self.get_logger().info(f"连接机械臂: {robot_ip}")
        self.rm_controller = RM_controller(robot_ip, rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.get_logger().info(f"连接成功，当前关节: {self.rm_controller.get_state()}")

        # MoveIt 初始化
        self.get_logger().info("初始化 MoveIt...")
        moveit_config = (
            MoveItConfigsBuilder(
                robot_name="rm_robot",
                package_name="rm_moveit2",
            )
            .robot_description(file_path="config/rm_75_6f_description.urdf.xacro")
            .trajectory_execution(file_path="config/moveit_controllers.yaml")
            .moveit_cpp(file_path="config/motion_planning_python_api_tutorial.yaml")
            .to_moveit_configs()
        )
        self.moveit = MoveItPy(node_name="moveit_py_node", config_dict=moveit_config.to_dict())
        self.group_name = group_name
        self.eef_link = eef_link
        self.arm = self.moveit.get_planning_component(self.group_name)
        self.robot_model: RobotModel = self.moveit.get_robot_model()
        self.jmg = self.robot_model.get_joint_model_group(self.group_name)
        self.joint_names = joint_names or self.jmg.active_joint_model_names
        if not self.joint_names:
            raise RuntimeError(f"无法从 JointModelGroup({self.group_name}) 获取关节名")

        # 关节状态发布（给 RViz / MoveIt 起始状态）
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / max(joint_state_hz, 0.1), self._publish_joint_state)

        # 订阅目标位姿
        self.pose_sub = self.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)
        self.currently_executing = False
        self.last_goal = None

        self.get_logger().info(f"监听位姿话题: {pose_topic}")
        self.get_logger().info(f"规划组: {self.group_name}, 末端链接: {self.eef_link}")

    # 检查角度值是否疑似为度
    def _looks_like_degree(self, joints: List[float]) -> bool:
        return any(abs(v) > 10.0 for v in joints)

    # 获取硬件关节值（自动检测度/弧度）
    def _get_hardware_joints_rad(self) -> List[float]:
        raw = list(self.rm_controller.get_state())
        if self._looks_like_degree(raw):
            if not hasattr(self, "_warned_deg"):
                self._warned_deg = True
                self.get_logger().info("检测到关节值疑似“度”，自动转弧度")
            return _deg2rad_list(raw)
        return raw

    # 发布关节状态
    def _publish_joint_state(self):
        joints_rad = self._get_hardware_joints_rad()
        size = min(len(joints_rad), len(self.joint_names))
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names[:size]
        msg.position = joints_rad[:size]
        self.joint_state_pub.publish(msg)

        # 同步 MoveIt start_state
        try:
            robot_state = RobotState(self.robot_model)
            robot_state.set_joint_group_active_positions(
                self.group_name, np.asarray(joints_rad[:size], dtype=float)
            )
            robot_state.update()
            # robot_state.enforce_bounds() # 强制关节限位（当前MoveIt2 Python API好像没有这个方法）
            ok = self.arm.set_start_state(robot_state=robot_state)
            if not ok:
                self.get_logger().warn("set_start_state 返回 False，请检查关节名/范围")
        except Exception as exc:
            self.get_logger().warn(f"同步 start_state 失败: {exc}")

    # ---- 回调与执行 ----
    # 检查位姿是否改变
    def _pose_changed(self, pose: PoseStamped) -> bool:
        p = pose.pose.position
        q = pose.pose.orientation
        current = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if self.last_goal is None:
            self.last_goal = current
            return True
        pos_eps = 1e-3
        ori_eps = 1e-3
        pos_changed = any(abs(a - b) > pos_eps for a, b in zip(current[:3], self.last_goal[:3]))
        ori_changed = any(abs(a - b) > ori_eps for a, b in zip(current[3:], self.last_goal[3:]))
        if pos_changed or ori_changed:
            self.last_goal = current
            return True
        return False

    # 位姿回调
    def _on_pose(self, msg: PoseStamped):
        if self.currently_executing:
            return
        if not self._pose_changed(msg):
            return
        self.currently_executing = True
        self.get_logger().info(
            f"收到目标位姿，frame={msg.header.frame_id} -> 规划并执行"
        )
        try:
            plan = self._plan_to_pose(msg)
            if plan:
                if self._execute_plan(plan):
                    self.get_logger().info("执行完成")
                else:
                    self.get_logger().error("执行失败")
            else:
                self.get_logger().error("规划失败")
        finally:
            self.currently_executing = False

    # 规划到指定位姿
    def _plan_to_pose(self, pose: PoseStamped):
        try:
            self._publish_joint_state()  # 确保起始状态最新
            self.arm.set_goal_state(pose_stamped_msg=pose, pose_link=self.eef_link)
            plan_result = self.arm.plan()
            if plan_result:
                self.get_logger().info("MoveIt 规划成功")
                return plan_result
            return None
        except Exception as exc:
            self.get_logger().error(f"规划出错: {exc}")
            return None

    # 执行规划轨迹  
    def _execute_plan(self, plan, group_joint_order=None) -> bool:
        if plan is None or plan.trajectory is None:
            self.get_logger().error("空规划或无轨迹")
            return False
        try:
            rt = plan.trajectory
            msg: RobotTrajectoryMsg = rt.get_robot_trajectory_msg()
            jt = msg.joint_trajectory
            name_to_index = {name: i for i, name in enumerate(jt.joint_names)}
            joint_order = group_joint_order or self.joint_names or jt.joint_names

            for i, point in enumerate(jt.points):
                positions = [point.positions[name_to_index[n]] for n in joint_order]
                positions_deg = _rad2deg_list(positions)
                ret = self.rm_controller.movej(positions_deg)
                if ret != 0:
                    self.get_logger().error(f"路点 {i} 执行失败，错误码: {ret}")
                    return False
            return True
        except Exception as exc:
            self.get_logger().error(f"执行轨迹出错: {exc}")
            return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="手动位姿 -> MoveIt 规划 -> RealMan 执行")
    parser.add_argument("--robot-ip", default="192.168.0.17", help="机械臂 IP")
    parser.add_argument("--pose-topic", default="/Current_OBJ_position_1", help="订阅的 PoseStamped 话题")
    parser.add_argument("--group", default="rm_robot_arm", help="MoveIt 规划组")
    parser.add_argument("--eef-link", default="Link7", help="末端链接名")
    parser.add_argument(
        "--joint-names",
        default="joint1,joint2,joint3,joint4,joint5,joint6,joint7",
        help="关节名顺序，逗号分隔（需与硬件状态顺序一致）",
    )
    parser.add_argument("--joint-state-hz", type=float, default=10.0, help="/joint_states 发布频率")
    args = parser.parse_args(argv)

    joint_names = [n.strip() for n in args.joint_names.split(",") if n.strip()]
    if not joint_names:
        print("关节名不能为空", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = SimpleMoveItController(
        robot_ip=args.robot_ip,
        pose_topic=args.pose_topic,
        group_name=args.group,
        eef_link=args.eef_link,
        joint_names=joint_names,
        joint_state_hz=args.joint_state_hz,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
