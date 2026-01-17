#!/usr/bin/env python3
from tarfile import tar_filter
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
from realman.RealMan import RM_controller
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e

class FoundationPoseMoveIt2Controller(Node):
    def __init__(self, object_ids, robot_ip="192.168.0.17", auto_move=False,
                 offset_z=0.16, enable_grasp=True, lift_height=0.2,
                 approach_distance=0.1, use_moveit=True):
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
                
                import os
                import yaml
                
                # # 检查配置文件是否存在
                # config_paths = {
                #     'urdf': "/home/ym/IML/FoundationPoseROS2/rm_moveit_config/src/rm_75_6f_description/urdf/rm_75_6f_description.urdf",
                #     'srdf': "/home/ym/IML/FoundationPoseROS2/rm_moveit_config/src/rm_moveit2/config/rm_75_6f_description.srdf",
                #     'controllers': "/home/ym/IML/FoundationPoseROS2/rm_moveit_config/src/rm_moveit2/config/moveit_controllers.yaml",
                #     'ompl': "/home/ym/IML/FoundationPoseROS2/rm_moveit_config/src/rm_moveit2/config/ompl_planning.yaml"
                # }
                
                # for name, path in config_paths.items():
                #     if os.path.exists(path):
                #         self.get_logger().info(f'✓ {name} 文件存在: {path}')
                #     else:
                #         self.get_logger().warn(f'✗ {name} 文件不存在: {path}')

                # # 构建 MoveIt 配置
                # moveit_config_builder = MoveItConfigsBuilder("rm_75_6f_description", package_name="rm_moveit2")
                # moveit_config_builder = moveit_config_builder.robot_description(
                #     file_path=config_paths['urdf']
                # )
                # moveit_config_builder = moveit_config_builder.robot_description_semantic(
                #     file_path=config_paths['srdf']
                # )
                # moveit_config_builder = moveit_config_builder.trajectory_execution(
                #     file_path=config_paths['controllers']
                # )
                
                # # 如果 ompl_planning.yaml 存在，添加它
                # if os.path.exists(config_paths['ompl']):
                #     self.get_logger().info('使用 ompl_planning.yaml 配置')
                #     moveit_config_builder = moveit_config_builder.planning_pipelines(
                #         pipelines=["ompl"],
                #         default_planning_pipeline="ompl"
                #     )
                # else:
                #     self.get_logger().warn('ompl_planning.yaml 不存在，使用默认配置')
                
                # moveit_config = moveit_config_builder.to_moveit_configs()
                # config_dict = moveit_config.to_dict()
                
                # # 打印完整配置结构用于调试
                # self.get_logger().info('=' * 60)
                # self.get_logger().info('配置字典结构:')
                # for key in config_dict.keys():
                #     value_type = type(config_dict[key]).__name__
                #     self.get_logger().info(f'  {key}: {value_type}')
                # self.get_logger().info('=' * 60)

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
                self.create_timer(0.5, self.sync_moveit_start_state)

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
        current_joints = list(self.rm_controller.get_state())

        # 这里假设 rm_controller.get_state() 返回的顺序与 joint_names 一致；
        # 若不一致，建议做一次名称-索引映射以重排
        if len(current_joints) != len(joint_names):
            self.get_logger().warn(
                f'当前关节数({len(current_joints)})与规划组({len(joint_names)})不一致，请确认映射关系'
            )

        # 发布 /joint_states 供可视化/TF
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = joint_names
        print(f"joint_names: {joint_names}")
        js.position = current_joints
        self.joint_state_pub.publish(js)

        # 同步 MoveIt 的 start_state
        robot_state = RobotState(self.robot_model)
        # robot_state.set_variable_positions(dict(zip(joint_names, current_joints)))
        robot_state.set_joint_group_active_positions(self.group_name, current_joints)
        robot_state.update()
        self.arm.set_start_state(robot_state)

    def convert_pose_to_realman_format(self, pose_msg, z_offset=0.0):
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z + self.offset_z + z_offset

        orientation = pose_msg.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        euler = R.from_quat(quat).as_euler('xyz', degrees=False)

        # target_pose = [x, y, z, 0, 1.57, 0]
        # test
        target_pose = [0.421877, 0.209509, 0.247668, 1.999, -0.067, -0.689]

        return target_pose

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
            self.get_logger().info(f'执行轨迹，包含 {len(jt.points)} 个路点')

            # 可选：按 joint_names 重新映射到控制器需要的顺序
            name_to_index = {name: i for i, name in enumerate(jt.joint_names)}
            if group_joint_order is None:
                # 如果你的控制器期望和规划组一致，可以改成：
                group_joint_order = self.joint_names or jt.joint_names

            for i, point in enumerate(jt.points):
                # point.positions 是与 jt.joint_names 对齐的 tuple
                positions = [point.positions[name_to_index[n]] for n in group_joint_order]
                result = self.rm_controller.movej(list(positions))
                if result != 0:
                    self.get_logger().error(f'执行路点 {i} 失败，错误码: {result}')
                    return False
                time.sleep(2)

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

            if self.use_moveit:
                success = self.move_with_moveit(approach_pose_stamped)
                if not success:
                    approach_pose = [x, y, approach_z, 0, 1.57, 0]
                    success = self.move_with_sdk(approach_pose)
            else:
                approach_pose = [x, y, approach_z, 0, 1.57, 0]
                success = self.move_with_sdk(approach_pose)

            if not success:
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

                if self.use_moveit:
                    success = self.move_with_moveit(retreat_pose_stamped)
                    if not success:
                        retreat_pose = [x, y, retreat_z, 0, 1.57, 0]
                        success = self.move_with_sdk(retreat_pose)
                else:
                    retreat_pose = [x, y, retreat_z, 0, 1.57, 0]
                    success = self.move_with_sdk(retreat_pose)

                if success:
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

            if self.enable_grasp:
                return self.grasp_and_lift_sequence(object_id, pose_msg)
            else:
                target_pose = self.convert_pose_to_realman_format(pose_msg)

                if self.use_moveit:
                    x, y, z = target_pose[0], target_pose[1], target_pose[2]
                    quat = [0.0, 0.707, 0.0, 0.707]
                    target_pose_stamped = self.create_pose_stamped(x, y, z, quat)

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

        print("\n" + "="*80)
        print("当前物体位姿与机械臂状态:")
        print("="*80)

        for obj_id in sorted(self.latest_poses.keys()):
            msg = self.latest_poses[obj_id]
            pos = msg.pose.position
            orient = msg.pose.orientation

            quat = [orient.x, orient.y, orient.z, orient.w]
            euler = R.from_quat(quat).as_euler('xyz', degrees=True)

            print(f"物体 {obj_id}:")
            print(f"  位置 (m): [{pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}]")
            print(f"  欧拉角 (°): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")

            target_pose = self.convert_pose_to_realman_format(msg)
            print(f"  机械臂目标: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]")
            print()

        if hasattr(self, 'rm_controller'):
            print(f"控制模式: {'MoveIt2规划' if self.use_moveit else 'SDK直接控制'}")
            print(f"夹爪状态: {self.get_gripper_status()}")
            print()

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
    parser.add_argument('--lift-height', type=float, default=0.2,
                       help='抓取后提升高度 (默认: 0.2m)')
    parser.add_argument('--approach-distance', type=float, default=0.1,
                       help='接近距离 (默认: 0.1m)')
    parser.add_argument('--no-moveit', action='store_true',
                       help='禁用MoveIt2，使用SDK直接控制')

    args = parser.parse_args()

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
            use_moveit=not args.no_moveit
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
