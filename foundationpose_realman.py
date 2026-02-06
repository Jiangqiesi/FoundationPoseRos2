#!/usr/bin/env python3
import math
from pathlib import Path
from tarfile import tar_filter

from warp import quat
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_model import RobotModel  # 注册 RobotModel 绑定
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.core.robot_state import RobotState
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
# from moveit.core.robot_state import RobotState
# from moveit.core.kinematic_constraints import construct_joint_constraint
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import time
import sys
import yaml
import trimesh
import trimesh.transformations as tra
from realman.RealMan import RM_controller
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e

class FoundationPoseMoveIt2Controller(Node):
    def __init__(self, object_ids, robot_ip="192.168.0.17", auto_move=False,
                 offset_z=0.16, enable_grasp=True, lift_height=0.2,
                 approach_distance=0.15, use_moveit=True,
                 grasp_file="demo_data/ship/ship3_grasp",
                 grasp_files=None,
                 pre_grasp_offset=0.10, place_pose=None,
                 place_approach_offset=0.08, home_pose=None):
        super().__init__('foundationpose_moveit2_controller')

        self.object_ids = object_ids
        self.auto_move = auto_move
        self.offset_z = offset_z
        self.enable_grasp = enable_grasp
        self.lift_height = lift_height
        self.approach_distance = approach_distance
        self.use_moveit = use_moveit

        self.latest_poses = {}
        self.subscribers = {}
        self.arm = None
        self.default_grasp_file = Path(grasp_file) if grasp_file else None
        self.default_grasp_library = self.load_grasp_file(self.default_grasp_file) if self.default_grasp_file else []
        # 与旧代码兼容，保留默认抓取库引用
        self.grasp_library = self.default_grasp_library
        self.object_grasp_files = {}
        self.object_grasp_libraries = {}
        self._warned_default_grasp = set()
        self._warned_missing_grasp = set()
        if grasp_files:
            for obj_id_raw, path in grasp_files.items():
                try:
                    obj_id = int(obj_id_raw)
                except ValueError:
                    self.get_logger().warn(f'抓取库配置的物体ID非法: {obj_id_raw}')
                    continue
                if obj_id not in self.object_ids:
                    self.get_logger().warn(f'抓取库配置的物体 {obj_id} 不在订阅列表 {self.object_ids}')
                file_path = Path(path)
                lib = self.load_grasp_file(file_path)
                self.object_grasp_files[obj_id] = file_path
                self.object_grasp_libraries[obj_id] = lib
                self.get_logger().info(f'物体 {obj_id} 使用专用抓取库: {file_path}')
        self.pre_grasp_offset = pre_grasp_offset
        self.place_pose = place_pose
        self.place_approach_offset = place_approach_offset
        self.home_pose = home_pose
        # # test不使用抓取库时，改为空列表
        # self.grasp_library = []
        
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        try:
            self.get_logger().info(f'正在连接机械臂: {robot_ip}')
            self.rm_controller = RM_controller(robot_ip, rm_thread_mode_e.RM_TRIPLE_MODE_E)
            initial_state = self.rm_controller.get_state()
            self.get_logger().info(f'机械臂连接成功，初始关节角度: {initial_state}')
        except Exception as e:
            self.get_logger().error(f'机械臂连接失败: {e}')
            sys.exit(1)

        if self.use_moveit:
            try:
                self.get_logger().info('正在初始化MoveIt2...')
                
                # 构建 MoveIt 配置
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

                # 初始化 MoveItPy
                self.get_logger().info('正在初始化 MoveItPy...')
                self.moveit = MoveItPy(
                    node_name="moveit_py_node", 
                    config_dict=moveit_config.to_dict()
                )
                
                self.get_logger().info('MoveIt2初始化成功')
                self.sync_moveit_start_state()
                # self.create_timer(0.5, self.sync_moveit_start_state)

                # 记录规划组名，便于后续统一使用
                self.group_name = "rm_robot_arm"  # 你现在就是用的这个名字
                self.arm = self.moveit.get_planning_component(self.group_name)
                self.robot_model = self.moveit.get_robot_model()

                # JointModelGroup
                self.jmg = self.robot_model.get_joint_model_group(self.group_name)

                self.joint_names = self.jmg.active_joint_model_names

                if not self.joint_names:
                    raise RuntimeError(f"无法从 JointModelGroup({self.group_name}) 获取关节名，请检查 SRDF/绑定。")

                # group_names = self.robot_model.get_joint_model_group_names()
                # self.get_logger().info(f'可用规划组: {group_names}')

                # preferred_groups = ["rm_robot_arm", "manipulator", "arm", "rm_robot"]
                # selected_group = next((g for g in preferred_groups if g in group_names), None)
                # if selected_group is None and group_names:
                #     selected_group = group_names[0]

                # if selected_group:
                #     try:
                #         self.arm = self.moveit.get_planning_component(selected_group)
                #         self.get_logger().info(f'使用规划组: {selected_group}')
                #     except Exception as arm_error:
                #         self.get_logger().error(f'获取规划组件失败: {arm_error}')
                #         self.use_moveit = False
                # else:
                #     self.get_logger().error('未找到可用的规划组，禁用 MoveIt2')
                #     self.use_moveit = False

                self.get_logger().info('MoveIt2初始化成功')

            except Exception as e:
                self.get_logger().error(f'MoveIt2初始化失败: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                self.get_logger().warn('将回退到直接SDK控制模式')
                self.use_moveit = False

        for obj_id in self.object_ids:
            topic_name = f'/Current_OBJ_position_{obj_id}'
            self.subscribers[obj_id] = self.create_subscription(
                PoseStamped,
                topic_name,
                lambda msg, id=obj_id: self.pose_callback(msg, id),
                10
            )
            self.get_logger().info(f'订阅物体 {obj_id} 位姿话题: {topic_name}')

        self.timer = self.create_timer(2.0, self.display_poses)

        mode_str = "MoveIt2运动规划" if self.use_moveit else "直接SDK控制"
        self.get_logger().info(f'控制器初始化完成 - 模式: {mode_str}')
        self.get_logger().info(f'自动移动: {"开启" if self.auto_move else "关闭"}')
        self.get_logger().info(f'抓取功能: {"开启" if self.enable_grasp else "关闭"}')

    def _deg2rad_list(self, vals):
        return [v * math.pi / 180.0 for v in vals]

    def _rad2deg_list(self, vals):
        return [v * 180.0 / math.pi for v in vals]

    def _normalize_with_model(self, joint_names, values_rad):
        """
        使用 MoveIt 的 RobotState + enforce_bounds() 来按模型界限归一化关节角。
        不直接访问 RobotModel 的 bounds 接口，避免绑定差异。
        """
        rs = RobotState(self.robot_model)

        # 逐个变量设置位置（弧度）
        for name, v in zip(joint_names, values_rad):
            # 建议做一次简单 2π wrap，避免极端大数（可选）
            # v = (v + math.pi) % (2.0 * math.pi) - math.pi
            rs.set_variable_position(name, float(v))

        rs.update()

        # 让 MoveIt 按 URDF/SRDF 的上下界修正
        try:
            rs.enforce_bounds()
        except Exception:
            # 某些绑定不抛异常，静默继续
            pass

        # 读回每个变量的值
        normalized = []
        for name in joint_names:
            # 某些绑定返回 numpy / array，取 float 即可
            normalized.append(float(rs.get_variable_position(name)))
        return normalized

    def get_robot_description(self):
        urdf_path = "/home/ym/IML/FoundationPoseROS2/rm_moveit_config/src/rm_75_6f_description/urdf/rm_75_6f_description.urdf"
        try:
            with open(urdf_path, 'r') as f:
                return f.read()
        except Exception as e:
            self.get_logger().error(f'读取URDF失败: {e}')
            return ""

    def get_srdf_content(self):
        srdf_path = "/home/ym/IML/FoundationPoseROS2/rm_moveit_config/src/rm_moveit2/config/rm_75_6f_description.srdf"
        try:
            with open(srdf_path, 'r') as f:
                return f.read()
        except Exception as e:
            self.get_logger().error(f'读取SRDF失败: {e}')
            return ""

    def load_grasp_file(self, grasp_path: Path):
        if grasp_path is None:
            self.get_logger().warn('未提供抓取姿态文件路径')
            return []
        if not isinstance(grasp_path, Path):
            grasp_path = Path(grasp_path)
        resolved_path = grasp_path if grasp_path.is_absolute() else Path(__file__).resolve().parent / grasp_path
        if not resolved_path.exists():
            self.get_logger().warn(f'抓取姿态文件不存在: {resolved_path}')
            return []
        try:
            with resolved_path.open('r') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f'读取抓取姿态失败: {e}')
            return []
        
        # 加载 raw 文件用于筛选
        grasp_path_raw = str(grasp_path) + "_raw.yml" if not str(grasp_path).endswith('.yml') else str(grasp_path).replace(".yml", "_raw.yml")
        resolved_path_raw = Path(grasp_path_raw) if Path(grasp_path_raw).is_absolute() else Path(__file__).resolve().parent / grasp_path_raw
        
        data_raw = {}
        if resolved_path_raw.exists():
            try:
                with resolved_path_raw.open('r') as f:
                    data_raw = yaml.safe_load(f) or {}
            except Exception as e:
                self.get_logger().error(f'读取抓取姿态(raw)失败: {e}')
        
        # 根据 raw 数据筛选出需要剔除的抓取名称
        excluded_grasps = set()
        raw_grasps = data_raw.get('grasps') or {}
        for name, entry in raw_grasps.items():
            pos = entry.get('position') or []
            if len(pos) >= 3:
                # 若 raw 中 z轴 < 0.02m，则剔除对应的抓取
                if pos[2] < 0.02 - 0.5:
                    excluded_grasps.add(name)
                    self.get_logger().info(f'根据 raw 数据筛选剔除抓取: {name} (y={pos[1]:.4f}, z={pos[2]:.4f})')

        grasps = []
        for name, entry in (data.get('grasps') or {}).items():
            # 如果该抓取在排除列表中，跳过
            if name in excluded_grasps:
                continue
                
            pos = entry.get('position') or []
            # # 位置偏移：x轴-0.035m, y轴偏移-0.02m, z轴
            # pos = [pos[0] - 0.030, pos[1] - 0.02, pos[2]]

            orient = entry.get('orientation') or {}
            xyz = orient.get('xyz') or []
            w = orient.get('w', None)
            if len(pos) != 3 or len(xyz) != 3 or w is None:
                self.get_logger().warn(f'抓取 {name} 数据不完整，已跳过')
                continue
            quat_xyzw = np.array([xyz[0], xyz[1], xyz[2], w], dtype=np.float64)
            grasps.append({
                "name": name,
                "confidence": float(entry.get('confidence', 0.0)),
                "position": np.array(pos, dtype=np.float64),
                "quat": quat_xyzw,
            })

        grasps.sort(key=lambda g: g["confidence"], reverse=True)
        self.get_logger().info(f'已从 {resolved_path} 载入 {len(grasps)} 个抓取候选 (筛除 {len(excluded_grasps)} 个)')
        return grasps

    def get_grasp_library_for_object(self, object_id):
        if object_id in self.object_grasp_libraries:
            lib = self.object_grasp_libraries[object_id]
            if not lib and object_id not in self._warned_missing_grasp:
                self.get_logger().warn(
                    f'物体 {object_id} 的抓取库为空: {self.object_grasp_files.get(object_id)}'
                )
                self._warned_missing_grasp.add(object_id)
            return lib

        if self.default_grasp_library:
            if object_id not in self._warned_default_grasp:
                self.get_logger().info(
                    f'物体 {object_id} 未配置专用抓取库，使用默认 {self.default_grasp_file}'
                )
                self._warned_default_grasp.add(object_id)
        else:
            if object_id not in self._warned_missing_grasp:
                self.get_logger().warn(
                    f'物体 {object_id} 未配置抓取库且未提供默认抓取库'
                )
                self._warned_missing_grasp.add(object_id)
        return self.default_grasp_library

    def pose_callback(self, msg, object_id):
        self.latest_poses[object_id] = msg

        if self.auto_move:
            self.move_to_object(object_id)

    def sync_moveit_start_state(self):
        if not self.use_moveit or self.arm is None:
            return

        # 使用在 __init__ 里取到的关节名
        joint_names = self.joint_names
        if not joint_names:
            self.get_logger().error('未能获取规划组关节名，跳过起始状态同步')
            return

        # 从实物控制器读取当前关节角
        raw_joints = list(self.rm_controller.get_state())  # 硬件读数，多半是“度”
        if len(raw_joints) != len(joint_names):
            self.get_logger().warn(
                f'当前关节数({len(raw_joints)})与规划组({len(joint_names)})不一致，请确认映射关系'
            )
        if any(abs(v) > 10 for v in raw_joints):
            self.get_logger().info('检测到硬件关节值疑似为“度”，将自动转换为弧度')

        # ——关键：度→弧度——
        joints_rad = self._deg2rad_list(raw_joints)

        # # 使用 RobotState + enforce_bounds() 做模型归一化
        # joints_rad = self._normalize_with_model(joint_names, joints_rad)
        # 发布 joint_states（注意：ROS 的惯例是“弧度”）
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = joint_names
        js.position = joints_rad
        self.joint_state_pub.publish(js)

        # 同步到 MoveIt 的 start_state（弧度）
        robot_state = RobotState(self.robot_model)
        # robot_state.set_variable_positions(dict(zip(joint_names, current_joints)))
        robot_state.set_joint_group_active_positions(self.group_name, np.asarray(joints_rad, dtype=float))
        robot_state.update()
        # self.arm.set_start_state(robot_state)
        try:
            # 再做一次边界检查/收敛，防止数值边界触发
            robot_state.enforce_bounds()
        except Exception:
            pass
        ok = self.arm.set_start_state(robot_state=robot_state)   # ← 关键修复
        if not ok:
            self.get_logger().warn('set_start_state() 返回 False，请检查关节名/范围/组名是否匹配')

    def convert_pose_to_realman_format(self, pose_msg, z_offset=0.05):
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z + self.offset_z + z_offset

        orientation = pose_msg.pose.orientation
        # # test
        quat_xyzw = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        # quat_xyzw = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float64)
        print(f"收到物体位姿: x={x:.4f}, y={y:.4f}, z={z:.4f}, "
              f"quat_xyzw={quat_xyzw}")

        rotation = R.from_quat(quat_xyzw)
        euler_rpy = rotation.as_euler('xyz', degrees=False)
        # z_axis_rotation = R.from_euler('z', np.pi / 2.0, degrees=False)  # 90° intrinsic rotation
        # rotated_rotation = rotation * z_axis_rotation  # apply around object-local z axis
        # euler_rpy = rotated_rotation.as_euler('xyz', degrees=False)

        # x_axis_rotation = R.from_euler('x', math.pi / 2.0, degrees=False)
        # rotated_rotation = rotation * x_axis_rotation
        # euler_rpy = rotated_rotation.as_euler('xyz', degrees=False)

        print(
            f"真实目标位姿: x={x:.4f}, y={y:.4f}, z={z:.4f}, "
            f"roll={euler_rpy[0]:.2f}, pitch={euler_rpy[1]:.2f}, yaw={euler_rpy[2]:.2f}"
        )
        target_pose = [x, y, z, euler_rpy[0], euler_rpy[1], euler_rpy[2]]
        
        # # test
        # target_pose = [x, y, z, euler_rpy[0], euler_rpy[1], euler_rpy[2]]

        return target_pose

    def transform_grasp_pose(self, obj_pos, obj_quat, grasp_pos_local, grasp_quat_local):
        r_obj = R.from_quat(obj_quat)
        T_world_obj = np.eye(4)
        T_world_obj[:3, :3] = r_obj.as_matrix()
        T_world_obj[:3, 3] = obj_pos

        r_grasp = R.from_quat(grasp_quat_local)
        T_obj_grasp = np.eye(4)
        T_obj_grasp[:3, :3] = r_grasp.as_matrix()
        T_obj_grasp[:3, 3] = grasp_pos_local

        T_world_grasp = T_world_obj @ T_obj_grasp
        final_pos = T_world_grasp[:3, 3]
        final_quat = R.from_matrix(T_world_grasp[:3, :3]).as_quat()
        return final_pos, final_quat

    def compute_pre_grasp_position(self, obj_pos, grasp_pos_world, offset):
        # direction = grasp_pos_world - obj_pos
        # norm = np.linalg.norm(direction)
        # if norm < 1e-6:
        #     direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        #     norm = 1.0
        # unit_dir = direction / norm
        # return grasp_pos_world + unit_dir * float(offset)
        # 让z轴抬升即可
        pre_grasp_pos = np.array(grasp_pos_world, dtype=np.float64)
        pre_grasp_pos[2] += float(offset)
        return pre_grasp_pos

    def create_pose_stamped(self, x, y, z, quat_xyzw):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = quat_xyzw[0]
        pose.pose.orientation.y = quat_xyzw[1]
        pose.pose.orientation.z = quat_xyzw[2]
        pose.pose.orientation.w = quat_xyzw[3]
        return pose

    def pose_list_to_pose_stamped(self, pose_list):
        """将[x, y, z, rx, ry, rz]列表转换为PoseStamped"""
        if pose_list is None or len(pose_list) != 6:
            raise ValueError("期望长度为6的放置/回零位姿")
        quat_xyzw = R.from_euler('xyz', pose_list[3:6], degrees=False).as_quat()
        return self.create_pose_stamped(
            float(pose_list[0]),
            float(pose_list[1]),
            float(pose_list[2]),
            quat_xyzw,
        )

    def pose_stamped_to_realman_pose(self, pose_stamped: PoseStamped):
        """将PoseStamped转换为RealMan SDK使用的[x, y, z, rx, ry, rz]格式"""
        pos = pose_stamped.pose.position
        orient = pose_stamped.pose.orientation
        quat_xyzw = np.array([orient.x, orient.y, orient.z, orient.w], dtype=np.float64)
        rpy = R.from_quat(quat_xyzw).as_euler('xyz', degrees=False)
        return [pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]]

    def generate_grasp_targets(self, pose_msg, grasp_library):
        if not grasp_library:
            return []

        pos = pose_msg.pose.position
        orient = pose_msg.pose.orientation
        obj_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        obj_quat = np.array([orient.x, orient.y, orient.z, orient.w], dtype=np.float64)

        targets = []
        for grasp in grasp_library:
            # # 先对抓取位姿进行变换，绕x轴旋转56度
            # base_rotation = R.from_quat(grasp["quat"])
            # additional_rotation = R.from_euler('x', math.radians(56), degrees=False)
            # combined_rotation = additional_rotation * base_rotation
            # modified_grasp_quat = combined_rotation.as_quat()

            final_pos, final_quat = self.transform_grasp_pose(
                obj_pos, obj_quat, grasp["position"], grasp["quat"]
            )
            pre_pos = self.compute_pre_grasp_position(obj_pos, final_pos, self.pre_grasp_offset)
            # final_pos = self.compute_pre_grasp_position(obj_pos, final_pos, 0.03)

            target_pose = self.create_pose_stamped(
                float(final_pos[0]),
                float(final_pos[1]),
                float(final_pos[2]),
                final_quat,
            )
            pre_pose = self.create_pose_stamped(
                float(pre_pos[0]),
                float(pre_pos[1]),
                float(pre_pos[2]),
                final_quat,
            )
            targets.append({
                "pose": target_pose,
                "pre_pose": pre_pose,
                "meta": grasp,
            })
        return targets

    def plan_to_pose(self, target_pose_stamped):
        if not self.use_moveit:
            return None
        try:
            self.sync_moveit_start_state()
            self.arm.set_goal_state(pose_stamped_msg=target_pose_stamped, pose_link="Link7")

            plan_result = self.arm.plan()

            if plan_result:
                self.get_logger().info('MoveIt2规划成功')
                return plan_result
            else:
                self.get_logger().warn('MoveIt2规划失败')
                return None

        except Exception as e:
            self.get_logger().error(f'MoveIt2规划出错: {e}')
            return None

    def execute_plan(self, plan, group_joint_order=None):
        if plan is None or plan.trajectory is None:
            self.get_logger().error('空规划或无轨迹')
            return False
        try:
            rt = plan.trajectory  # moveit.core.robot_trajectory.RobotTrajectory
            msg: RobotTrajectoryMsg = rt.get_robot_trajectory_msg()  # 转消息
            jt = msg.joint_trajectory                      # trajectory_msgs/JointTrajectory
            total_points = len(jt.points)
            max_points = 10
            points_to_execute = jt.points

            if total_points > max_points:
                # 均匀采样路点，确保首尾包含在内
                indices = [
                    int(i * (total_points - 1) / (max_points - 1))
                    for i in range(max_points - 1)
                ]
                indices.append(total_points - 1)
                points_to_execute = [jt.points[idx] for idx in indices]
                self.get_logger().info(
                    f'轨迹路点从 {total_points} 个均匀采样至 {len(points_to_execute)} 个'
                )
            else:
                self.get_logger().info(f'执行轨迹，包含 {total_points} 个路点')

            # 可选：按 joint_names 重新映射到控制器需要的顺序
            name_to_index = {name: i for i, name in enumerate(jt.joint_names)}
            if group_joint_order is None:
                # 如果你的控制器期望和规划组一致，可以改成：
                group_joint_order = self.joint_names or jt.joint_names

            for i, point in enumerate(points_to_execute):
                # point.positions 是与 jt.joint_names 对齐的 tuple
                positions = [point.positions[name_to_index[n]] for n in group_joint_order]
                # positions_stamped 是按 group_joint_order 排好序、单位=弧度 的序列
                positions_deg = self._rad2deg_list(positions)
                result = self.rm_controller.movej(positions_deg)
                if result != 0:
                    self.get_logger().error(f'执行路点 {i} 失败，错误码: {result}')
                    return False
                # time.sleep(1)

            self.get_logger().info('轨迹执行完成')
            return True
        except Exception as e:
            self.get_logger().error(f'执行轨迹时出错: {e}')
            return False

    def move_with_moveit(self, target_pose_stamped):
        self.get_logger().info('使用MoveIt2规划运动...')

        plan = self.plan_to_pose(target_pose_stamped)

        if plan:
            return self.execute_plan(plan)
        else:
            self.get_logger().warn('MoveIt2规划失败，回退到直接控制')
            return False
    
    def execute_pose_with_fallback(self, pose_stamped: PoseStamped, step_name: str = ""):
        """优先使用MoveIt执行，失败则转换为SDK姿态直接发送"""
        desc = f"({step_name})" if step_name else ""
        if self.use_moveit:
            if self.move_with_moveit(pose_stamped):
                return True
            self.get_logger().warn(f'MoveIt2执行{desc}失败，尝试SDK直控')

        target_pose = self.pose_stamped_to_realman_pose(pose_stamped)
        return self.move_with_sdk(target_pose)

    def try_moveit_grasp_candidates(self, object_id, pose_msg, grasp_library):
        if not self.use_moveit:
            return False

        candidates = self.generate_grasp_targets(pose_msg, grasp_library)
        if not candidates:
            self.get_logger().warn(f'物体 {object_id} 未找到抓取候选位姿，跳过组合规划')
            return False

        total = len(candidates)
        for idx, candidate in enumerate(candidates, start=1):
            meta = candidate["meta"]
            pre = candidate["pre_pose"].pose.position
            p = candidate["pose"].pose.position
            # 对抓取候选进行筛选，若z轴过低则跳过
            if p.z < 0.04:
                self.get_logger().info(
                    f'跳过抓取候选 {meta["name"]}，z轴过低 (z={p.z:.3f}m)'
                )
                continue
            
            self.get_logger().info(
                f'尝试抓取候选 {idx}/{total} ({meta["name"]}, conf={meta["confidence"]:.3f})\n'
                f'  预抓取: ({pre.x:.3f}, {pre.y:.3f}, {pre.z:.3f}) -> 目标: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
            )

            if self.execute_grasp_candidate(candidate):
                self.get_logger().info(f'候选 {meta["name"]} 抓取完成')
                return True

            self.get_logger().warn(f'候选 {meta["name"]} 抓取失败，尝试下一个')

        self.get_logger().error('所有抓取候选抓取均失败')
        return False

    def execute_grasp_candidate(self, candidate):
        """使用生成的抓取位姿直接完成 预抓取-抓取-提升 流程，避免重复生成目标"""
        meta = candidate.get("meta", {})
        pre_pose = candidate["pre_pose"]
        grasp_pose = candidate["pose"]

        self.get_logger().info(f'使用抓取候选 {meta.get("name", "unknown")} 执行抓取序列')
        # 确保夹爪张开
        self.rm_controller.set_gripper(1.0)
        time.sleep(0.5)

        # 1. 预抓取
        if not self.execute_pose_with_fallback(pre_pose, "预抓取"):
            return False

        # 2. 移动到抓取姿态
        if not self.execute_pose_with_fallback(grasp_pose, "抓取位姿"):
            return False

        # 3. 闭合夹爪
        current_gripper_pos = self.rm_controller.get_gripper()
        print(f'抓取前夹爪位置: {current_gripper_pos:.3f}')
        self.rm_controller.set_gripper(-1.0) # 假设0.1表示张开10%
        time.sleep(2.0)
        new_gripper_pos = self.rm_controller.get_gripper()
        print(f'抓取后夹爪位置: {new_gripper_pos:.3f}')

        if abs(new_gripper_pos - current_gripper_pos) <= 0.05:
            self.get_logger().warn('抓取失败，夹爪位置变化不明显')
            self.release_object()
            self.execute_pose_with_fallback(lift_pose, "提升")
            return False

        # 4. 提升物体，保持同一姿态
        gpos = grasp_pose.pose.position
        gquat = grasp_pose.pose.orientation
        lift_pose = self.create_pose_stamped(
            float(gpos.x),
            float(gpos.y),
            float(gpos.z + self.lift_height),
            [gquat.x, gquat.y, gquat.z, gquat.w],
        )

        if not self.execute_pose_with_fallback(lift_pose, "提升"):
            self.release_object()
            return False

        time.sleep(2.0)
        return True

    def move_with_sdk(self, target_pose):
        self.get_logger().info(f'使用SDK直接移动到: {target_pose}')
        result = self.rm_controller.movel(target_pose)

        if result == 0:
            self.get_logger().info('SDK移动成功')
            return True
        else:
            self.get_logger().error(f'SDK移动失败，错误码: {result}')
            return False

    def grasp_and_lift_sequence(self, object_id, pose_msg):
        try:
            self.get_logger().info(f'开始三段式抓取序列: 物体 {object_id}')

            self.rm_controller.set_gripper(1.0)
            time.sleep(1.0)

            x = pose_msg.pose.position.x
            y = pose_msg.pose.position.y
            z = pose_msg.pose.position.z

            approach_z = z + self.offset_z + self.approach_distance
            grasp_z = z + self.offset_z
            retreat_z = grasp_z + self.lift_height

            quat = [0.0, 0.707, 0.0, 0.707]

            self.get_logger().info(f'第1步: 移动到接近点 (z={approach_z:.3f}m)')
            approach_pose_stamped = self.create_pose_stamped(x, y, approach_z, quat)

            if not self.execute_pose_with_fallback(approach_pose_stamped, "接近"):
                self.get_logger().error('接近阶段失败')
                self.release_object()
                return False

            time.sleep(0.5)

            self.get_logger().info(f'第2步: 下降到抓取点 (z={grasp_z:.3f}m)')
            grasp_pose = [x, y, grasp_z, 0, 1.57, 0]
            success = self.move_with_sdk(grasp_pose)

            if not success:
                self.get_logger().error('下降阶段失败')
                self.release_object()
                return False

            time.sleep(0.5)

            self.get_logger().info('第3步: 闭合夹爪抓取')
            current_gripper_pos = self.rm_controller.get_gripper()
            self.get_logger().info(f'抓取前夹爪位置: {current_gripper_pos:.3f}')

            self.rm_controller.set_gripper(-0.9)
            time.sleep(2.0)

            new_gripper_pos = self.rm_controller.get_gripper()
            self.get_logger().info(f'抓取后夹爪位置: {new_gripper_pos:.3f}')

            if abs(new_gripper_pos - current_gripper_pos) > 0.05 and new_gripper_pos > 0.05:
                self.get_logger().info('抓取成功')

                self.get_logger().info(f'第4步: 提升物体 (z={retreat_z:.3f}m)')
                retreat_pose_stamped = self.create_pose_stamped(x, y, retreat_z, quat)

                if self.execute_pose_with_fallback(retreat_pose_stamped, "提升"):
                    self.get_logger().info(f'成功完成物体 {object_id} 的抓取和提升')
                    time.sleep(5.0)
                    return True
                else:
                    self.get_logger().error('提升阶段失败')
                    self.release_object()
                    return False
            else:
                self.get_logger().warn('抓取失败，夹爪位置变化不明显')
                self.release_object()
                return False

        except Exception as e:
            self.get_logger().error(f'抓取序列出错: {e}')
            self.release_object()
            return False

    def release_object(self):
        try:
            self.get_logger().info('正在释放物体...')
            self.rm_controller.set_gripper(0.8)
            time.sleep(1.0)
        except Exception as e:
            self.get_logger().error(f'释放物体时出错: {e}')

    def place_sequence(self):
        if self.place_pose is None:
            self.get_logger().info('未配置放置位姿，跳过放置动作')
            return True

        if len(self.place_pose) != 6:
            self.get_logger().error('放置位姿需提供6个元素 [x y z rx ry rz]')
            return False

        try:
            target = list(self.place_pose)
            rpy = target[3:6]
            pre_target = [
                target[0],
                target[1],
                target[2] + float(self.place_approach_offset),
                rpy[0],
                rpy[1],
                rpy[2],
            ]

            pre_pose_stamped = self.pose_list_to_pose_stamped(pre_target)
            place_pose_stamped = self.pose_list_to_pose_stamped(target)

            if not self.execute_pose_with_fallback(pre_pose_stamped, "放置-上方"):
                return False

            if not self.execute_pose_with_fallback(place_pose_stamped, "放置-下降"):
                return False

            self.release_object()
            time.sleep(0.5)

            # 放置完成后抬回上方，提高与其他动作切换的安全性
            self.execute_pose_with_fallback(pre_pose_stamped, "放置-抬升")
            return True
        except Exception as e:
            self.get_logger().error(f'放置任务出错: {e}')
            return False

    def move_to_home_pose(self):
        if self.home_pose is None:
            return True

        if len(self.home_pose) != 6:
            self.get_logger().error('初始位姿需提供6个元素 [x y z rx ry rz]')
            return False

        try:
            home_pose_stamped = self.pose_list_to_pose_stamped(self.home_pose)
            target_pose = self.pose_stamped_to_realman_pose(home_pose_stamped)
            return self.move_with_sdk(target_pose)
        except Exception as e:
            self.get_logger().error(f'回初始位姿失败: {e}')
            return False

    def post_grasp_actions(self):
        if not self.place_sequence():
            return False
        return self.move_to_home_pose()

    def get_gripper_status(self):
        try:
            pos = self.rm_controller.get_gripper()
            status = '张开' if pos > 0.5 else '闭合' if pos < 0.2 else '半开'
            return f"夹爪位置: {pos:.3f} ({status})"
        except Exception as e:
            return f"获取夹爪状态失败: {e}"

    def move_to_object(self, object_id):
        if object_id not in self.latest_poses:
            self.get_logger().warn(f'未找到物体 {object_id} 的位姿信息')
            return False

        try:
            pose_msg = self.latest_poses[object_id]

            grasp_library = self.get_grasp_library_for_object(object_id)

            if self.use_moveit and grasp_library:
                grasp_success = self.try_moveit_grasp_candidates(object_id, pose_msg, grasp_library)
                if grasp_success:
                    return self.post_grasp_actions()
                else:
                    self.get_logger().warn('组合抓取规划失败，回退到原有流程')

            if self.enable_grasp:
                # TODO: fix with condition of grasp pose 
                if self.grasp_and_lift_sequence(object_id, pose_msg):
                    return self.post_grasp_actions()
                return False
            else:
                target_pose = self.convert_pose_to_realman_format(pose_msg)

                if self.use_moveit:
                    x, y, z = target_pose[0], target_pose[1], target_pose[2]
                    # RealMan接口使用的姿态是Roll/Pitch/Yaw，需要转成四元数给MoveIt
                    rpy = np.array(target_pose[3:6], dtype=np.float64) if len(target_pose) >= 6 else np.zeros(3, dtype=np.float64)
                    base_rotation = R.from_euler('xyz', rpy, degrees=False)
                    # # MoveIt的末端坐标系相对RealMan SDK存在约90°的绕x轴偏差，提前补偿
                    # moveit_compensation = R.from_euler('x', math.pi / 2.0, degrees=False)
                    # compensated_rotation = moveit_compensation * base_rotation
                    # quat_xyzw = compensated_rotation.as_quat()
                    quat_xyzw = base_rotation.as_quat()
                    target_pose_stamped = self.create_pose_stamped(x, y, z, quat_xyzw)

                    success = self.move_with_moveit(target_pose_stamped)
                    if not success:
                        success = self.move_with_sdk(target_pose)
                    return success
                else:
                    return self.move_with_sdk(target_pose)

        except Exception as e:
            self.get_logger().error(f'移动到物体时出错: {e}')
            return False

    def display_poses(self):
        if not self.latest_poses:
            return

        # print("\n" + "="*80)
        # print("当前物体位姿与机械臂状态:")
        # print("="*80)

        # for obj_id in sorted(self.latest_poses.keys()):
        #     msg = self.latest_poses[obj_id]
        #     pos = msg.pose.position
        #     orient = msg.pose.orientation

        #     quat = [orient.x, orient.y, orient.z, orient.w]
        #     euler = R.from_quat(quat).as_euler('xyz', degrees=True)

        #     print(f"物体 {obj_id}:")
        #     print(f"  位置 (m): [{pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}]")
        #     print(f"  欧拉角 (°): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")

        #     target_pose = self.convert_pose_to_realman_format(msg)
        #     print(f"  机械臂目标: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]")
        #     print()

        # if hasattr(self, 'rm_controller'):
        #     print(f"控制模式: {'MoveIt2规划' if self.use_moveit else 'SDK直接控制'}")
        #     print(f"夹爪状态: {self.get_gripper_status()}")
        #     print()

    def manual_control_loop(self):
        print("\n手动控制模式:")
        mode_str = "MoveIt2规划移动并抓取" if (self.use_moveit and self.enable_grasp) else \
                   "MoveIt2规划移动" if self.use_moveit else \
                   "直接移动并抓取" if self.enable_grasp else "直接移动"
        print(f"- 输入物体ID (如: 1) {mode_str}")
        print("- 输入 'status' 查看当前状态")
        print("- 输入 'gripper open' 张开夹爪")
        print("- 输入 'gripper close' 闭合夹爪")
        print("- 输入 'release' 释放物体")
        print("- 输入 'toggle' 切换控制模式")
        print("- 输入 'q' 退出")

        while rclpy.ok():
            try:
                user_input = input("\n请输入命令 > ").strip()

                if user_input.lower() in ['q', 'quit']:
                    print("退出程序")
                    break

                elif user_input.lower() == 'status':
                    self.display_poses()
                    continue

                elif user_input.lower() == 'gripper open':
                    self.rm_controller.set_gripper(0.8)
                    print("夹爪已张开")
                    continue

                elif user_input.lower() == 'gripper close':
                    self.rm_controller.set_gripper(-0.8)
                    print("夹爪已闭合")
                    continue

                elif user_input.lower() == 'release':
                    self.release_object()
                    continue

                elif user_input.lower() == 'toggle':
                    self.use_moveit = not self.use_moveit
                    mode = "MoveIt2规划" if self.use_moveit else "SDK直接控制"
                    print(f"已切换到: {mode}")
                    continue

                try:
                    obj_id = int(user_input)
                    if obj_id in self.object_ids:
                        success = self.move_to_object(obj_id)
                        if success:
                            print(f"成功操作物体 {obj_id}")
                        else:
                            print(f"操作物体 {obj_id} 失败")
                    else:
                        print(f"物体ID {obj_id} 不在订阅列表中: {self.object_ids}")

                except ValueError:
                    print("输入错误，请输入有效的物体ID或命令")

            except KeyboardInterrupt:
                print("\n收到退出信号")
                break
            except Exception as e:
                print(f"发生错误: {e}")

    def cleanup(self):
        try:
            if hasattr(self, 'rm_controller'):
                self.get_logger().info("正在断开机械臂连接...")
                del self.rm_controller
        except Exception as e:
            self.get_logger().error(f"清理资源时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='FoundationPose与MoveIt2集成控制')
    parser.add_argument('--objects', nargs='+', type=int, default=[1],
                       help='要订阅的物体ID列表 (默认: [1])')
    parser.add_argument('--robot-ip', type=str, default='192.168.0.17',
                       help='机械臂IP地址 (默认: 192.168.0.17)')
    parser.add_argument('--auto-move', action='store_true',
                       help='启用自动移动模式')
    parser.add_argument('--offset-z', type=float, default=0.16,
                       help='Z方向偏移量 (默认: 0.16m)')
    parser.add_argument('--disable-grasp', action='store_true',
                       help='禁用抓取功能')
    parser.add_argument('--lift-height', type=float, default=0.1,
                       help='抓取后提升高度 (默认: 0.1m)')
    parser.add_argument('--approach-distance', type=float, default=0.1,
                       help='接近距离 (默认: 0.1m)')
    parser.add_argument('--grasp-file', type=str, default='demo_data/ship_data/test_grasp_q2',
                       help='抓取姿态文件 (默认: demo_data/ship_data/test_grasp_q2)')
    parser.add_argument('--grasp-files', nargs='*', default=[],
                       help='为特定物体指定抓取姿态文件，格式 <obj_id>:<path>，例如 1:demo_data/a.yml 2:demo_data/b.yml')
    parser.add_argument('--pre-grasp-offset', type=float, default=0.05,
                       help='预抓取点相对抓取点沿物体方向外移距离 (默认: 0.05m)')
    parser.add_argument('--place-pose', nargs=6, type=float,
                       help='放置点6D位姿 [x y z rx ry rz] (单位: m/rad)')
    parser.add_argument('--place-approach-offset', type=float, default=0.08,
                       help='放置前相对目标的上方偏移高度 (默认: 0.08m)')
    parser.add_argument('--home-pose', nargs=6, type=float,
                       help='放置完成后回到的初始位姿 [x y z rx ry rz] (单位: m/rad)')
    parser.add_argument('--no-moveit', action='store_true',
                       help='禁用MoveIt2，使用SDK直接控制')

    args = parser.parse_args()

    grasp_file_map = {}
    for item in args.grasp_files or []:
        if ':' not in item:
            print(f"跳过无效的 --grasp-files 参数: {item}，格式应为 <obj_id>:<path>")
            continue
        obj_str, path = item.split(':', 1)
        try:
            obj_id = int(obj_str)
        except ValueError:
            print(f"跳过无效的物体ID: {obj_str}")
            continue
        grasp_file_map[obj_id] = path

    rclpy.init()

    controller = None
    try:
        controller = FoundationPoseMoveIt2Controller(
            object_ids=args.objects,
            robot_ip=args.robot_ip,
            auto_move=args.auto_move,
            offset_z=args.offset_z,
            enable_grasp=not args.disable_grasp,
            lift_height=args.lift_height,
            approach_distance=args.approach_distance,
            use_moveit=not args.no_moveit,
            grasp_file=args.grasp_file,
            grasp_files=grasp_file_map,
            pre_grasp_offset=args.pre_grasp_offset,
            place_pose=args.place_pose,
            place_approach_offset=args.place_approach_offset,
            home_pose=args.home_pose
        )

        print(f"开始监听物体 {args.objects} 的位姿信息...")
        print("按 Ctrl+C 退出")

        if args.auto_move:
            rclpy.spin(controller)
        else:
            import threading

            ros_thread = threading.Thread(target=lambda: rclpy.spin(controller))
            ros_thread.daemon = True
            ros_thread.start()

            controller.manual_control_loop()

    except KeyboardInterrupt:
        print("\n收到退出信号，正在关闭...")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        if controller:
            controller.cleanup()
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
