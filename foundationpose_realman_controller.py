#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import time
import sys
from realman.RealMan import RM_controller
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e

class FoundationPoseRealmanController(Node):
    def __init__(self, object_ids, robot_ip="192.168.0.17", auto_move=False, offset_z=0.16, enable_grasp=True, lift_height=0.2, down_z=0.1):
        """初始化FoundationPose与Realman机械臂控制器"""
        super().__init__('foundationpose_realman_controller')

        # 参数
        self.object_ids = object_ids
        self.auto_move = auto_move
        self.offset_z = offset_z  # Z方向偏移量，避免直接接触物体
        self.enable_grasp = enable_grasp  # 是否启用抓取功能
        self.lift_height = lift_height  # 抓取后向上移动的高度(m)
        self.down_z = down_z  # 抓取前向下移动的高度(m)

        # 存储最新位姿
        self.latest_poses = {}
        self.subscribers = {}

        # 初始化机械臂
        try:
            self.get_logger().info(f'正在连接机械臂: {robot_ip}')
            self.rm_controller = RM_controller(robot_ip, rm_thread_mode_e.RM_TRIPLE_MODE_E)
            initial_state = self.rm_controller.get_state()
            self.get_logger().info(f'机械臂连接成功，初始关节角度: {initial_state}')
        except Exception as e:
            self.get_logger().error(f'机械臂连接失败: {e}')
            sys.exit(1)

        # 为每个物体创建订阅者
        for obj_id in self.object_ids:
            topic_name = f'/Current_OBJ_position_{obj_id}'
            self.subscribers[obj_id] = self.create_subscription(
                PoseStamped,
                topic_name,
                lambda msg, id=obj_id: self.pose_callback(msg, id),
                10
            )
            self.get_logger().info(f'订阅物体 {obj_id} 位姿话题: {topic_name}')

        # 创建定时器定期显示位姿
        self.timer = self.create_timer(2.0, self.display_poses)

        self.get_logger().info('FoundationPose-Realman控制器初始化完成')
        self.get_logger().info(f'自动移动模式: {"开启" if auto_move else "关闭"}')
        self.get_logger().info(f'抓取功能: {"开启" if enable_grasp else "关闭"}')
        if enable_grasp:
            self.get_logger().info(f'抓取后提升高度: {lift_height*1000:.0f}mm')
        if not auto_move:
            self.get_logger().info('手动模式: 输入物体ID移动并抓取物体，输入"q"退出')

    def pose_callback(self, msg, object_id):
        """位姿回调函数"""
        self.latest_poses[object_id] = msg

        if self.auto_move:
            # 自动移动模式：收到位姿后立即移动
            self.move_to_object(object_id)

    def convert_pose_to_realman_format(self, pose_msg):
        """将ROS位姿消息转换为Realman机械臂格式"""
        # 提取位置 (m)
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z + self.offset_z + self.down_z  # 添加Z方向偏移和抓取前下移高度

        # 提取四元数并转换为欧拉角 (rad)
        orientation = pose_msg.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        euler = R.from_quat(quat).as_euler('xyz', degrees=False)  # 弧度制

        # Realman格式: [x, y, z, rx, ry, rz]
        target_pose = [x, y, z, 0, 1.57, 0]

        return target_pose

    def grasp_and_lift_sequence(self, object_id, grasp_pose):
        """执行抓取和提升序列"""
        try:
            # self.get_logger().info(f'开始抓取物体 {object_id}')

            # 向下移动到抓取位置
            down_pose = grasp_pose.copy()
            down_pose[2] -= self.down_z  # Z方向向下移动
            self.get_logger().info(f'正在移动到位置: {down_pose}')
            down_result = self.rm_controller.movel(down_pose)

            if down_result != 0:
                self.get_logger().error(f'接近失败，错误码: {down_result}')
                # 下降失败时释放物体
                self.release_object()
                return False
            
            # 等待机械臂稳定
            time.sleep(0.5)

            # 获取当前夹爪状态
            current_gripper_pos = self.rm_controller.get_gripper()
            self.get_logger().info(f'当前夹爪位置: {current_gripper_pos:.3f}')

            # 闭合夹爪抓取物体
            # 使用增量控制，向闭合方向移动
            self.get_logger().info('正在闭合夹爪抓取物体...')
            self.rm_controller.set_gripper(-0.9)  # 负值表示闭合

            # 等待夹爪动作完成
            time.sleep(2.0)

            # 检查夹爪是否成功抓取
            new_gripper_pos = self.rm_controller.get_gripper()
            self.get_logger().info(f'抓取后夹爪位置: {new_gripper_pos:.3f}')

            # 简单的抓取成功检测：夹爪位置变化且不为完全闭合
            # if True:
            if abs(new_gripper_pos - current_gripper_pos) > 0.05 and new_gripper_pos > 0.05:
                self.get_logger().info('抓取成功，开始向上提升')

                # 计算提升后的位置
                lift_pose = down_pose.copy()
                lift_pose[2] += self.lift_height # Z方向向上移动

                self.get_logger().info(f'正在提升到位置: {lift_pose}')

                # 执行提升动作
                lift_result = self.rm_controller.movel(lift_pose)

                if lift_result == 0:
                    self.get_logger().info(f'成功完成物体 {object_id} 的抓取和提升')
                    time.sleep(10)
                    return True
                else:
                    self.get_logger().error(f'提升失败，错误码: {lift_result}')
                    # 提升失败时释放物体
                    self.release_object()
                    return False
            else:
                self.get_logger().warn('抓取可能失败，夹爪位置变化不明显')
                # 释放夹爪
                self.release_object()
                return False

        except Exception as e:
            self.get_logger().error(f'抓取和提升过程中发生错误: {e}')
            # 发生错误时尝试释放物体
            self.release_object()
            return False

    def release_object(self):
        """释放夹爪中的物体"""
        try:
            self.get_logger().info('正在释放物体...')
            self.rm_controller.set_gripper(0.8)  # 正值表示张开
            time.sleep(1.0)
        except Exception as e:
            self.get_logger().error(f'释放物体时出错: {e}')

    def get_gripper_status(self):
        """获取夹爪状态信息"""
        try:
            pos = self.rm_controller.get_gripper()
            return f"夹爪位置: {pos:.3f} ({'张开' if pos > 0.5 else '闭合' if pos < 0.2 else '半开'})"
        except Exception as e:
            return f"获取夹爪状态失败: {e}"

    def move_to_object(self, object_id):
        """移动机械臂到指定物体位置"""
        if object_id not in self.latest_poses:
            self.get_logger().warn(f'未找到物体 {object_id} 的位姿信息')
            return False

        try:
            self.get_logger().info('正在松开夹爪抓取物体...')
            self.rm_controller.set_gripper(1)  # 负值表示闭合
            # 转换位姿格式
            target_pose = self.convert_pose_to_realman_format(self.latest_poses[object_id])

            self.get_logger().info(f'正在移动到物体 {object_id} 位置: {target_pose}')

            # 执行移动
            result = self.rm_controller.movel(target_pose)

            if result == 0:
                self.get_logger().info(f'成功移动到物体 {object_id} 位置')

                # 如果启用抓取功能，则执行抓取序列
                if self.enable_grasp:
                    return self.grasp_and_lift_sequence(object_id, target_pose)
                return True
            else:
                self.get_logger().error(f'移动失败，错误码: {result}')
                return False

        except Exception as e:
            self.get_logger().error(f'移动过程中发生错误: {e}')
            return False

    def display_poses(self):
        """显示当前所有物体位姿"""
        if not self.latest_poses:
            return

        print("\n" + "="*80)
        print("当前物体位姿与机械臂状态:")
        print("="*80)

        for obj_id in sorted(self.latest_poses.keys()):
            msg = self.latest_poses[obj_id]
            pos = msg.pose.position
            orient = msg.pose.orientation

            # 转换到欧拉角
            quat = [orient.x, orient.y, orient.z, orient.w]
            euler = R.from_quat(quat).as_euler('xyz', degrees=True)

            print(f"物体 {obj_id}:")
            print(f"  位置 (m): [{pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}]")
            print(f"  欧拉角 (°): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
            print(f"  欧拉角 (rad): [{np.deg2rad(euler[0]):.4f}, {np.deg2rad(euler[1]):.4f}, {np.deg2rad(euler[2]):.4f}]")

            # 显示机械臂目标位置（加上偏移）
            target_pose = self.convert_pose_to_realman_format(msg)
            print(f"  机械臂目标: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]")
            print()

        # 显示夹爪状态
        if hasattr(self, 'rm_controller'):
            print(f"夹爪状态: {self.get_gripper_status()}")
            print()

    def manual_control_loop(self):
        """手动控制循环"""
        print("\n手动控制模式:")
        print(f"- 输入物体ID (如: 1) {'移动并抓取物体' if self.enable_grasp else '移动到该物体位置'}")
        print("- 输入 'status' 查看当前状态")
        print("- 输入 'gripper open' 张开夹爪")
        print("- 输入 'gripper close' 闭合夹爪")
        print("- 输入 'release' 释放物体")
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

                try:
                    obj_id = int(user_input)
                    if obj_id in self.object_ids:
                        success = self.move_to_object(obj_id)
                        if success:
                            action = "抓取并提升" if self.enable_grasp else "移动到"
                            print(f"已成功{action}物体 {obj_id}")
                        else:
                            action = "抓取" if self.enable_grasp else "移动到"
                            print(f"{action}物体 {obj_id} 失败")
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
        """清理资源"""
        try:
            if hasattr(self, 'rm_controller'):
                self.get_logger().info("正在断开机械臂连接...")
                del self.rm_controller
        except Exception as e:
            self.get_logger().error(f"清理资源时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='FoundationPose与Realman机械臂集成控制')
    parser.add_argument('--objects', nargs='+', type=int, default=[1],
                       help='要订阅的物体ID列表 (默认: [1])')
    parser.add_argument('--robot-ip', type=str, default='192.168.0.17',
                       help='机械臂IP地址 (默认: 192.168.0.17)')
    parser.add_argument('--auto-move', action='store_true',
                       help='启用自动移动模式（收到位姿后立即移动）')
    parser.add_argument('--offset-z', type=float, default=0.16,
                       help='Z方向偏移量，避免碰撞 (默认: 0.16m)')
    parser.add_argument('--disable-grasp', action='store_true',
                       help='禁用抓取功能，仅移动到位置')
    parser.add_argument('--lift-height', type=float, default=0.1,
                       help='抓取后向上提升的高度 (默认: 0.1m = 100mm)')
    parser.add_argument('--down-z', type=float, default=0.05,
                        help='抓取前向下移动的高度 (默认: 0.05m = 50mm)')

    args = parser.parse_args()

    rclpy.init()

    controller = None
    try:
        controller = FoundationPoseRealmanController(
            object_ids=args.objects,
            robot_ip=args.robot_ip,
            auto_move=args.auto_move,
            offset_z=args.offset_z,
            enable_grasp=not args.disable_grasp,
            lift_height=args.lift_height,
            down_z=args.down_z
        )

        print(f"开始监听物体 {args.objects} 的位姿信息...")
        print("按 Ctrl+C 退出")

        if args.auto_move:
            # 自动模式：持续运行ROS节点
            rclpy.spin(controller)
        else:
            # 手动模式：在单独线程中运行ROS，主线程处理用户输入
            import threading

            ros_thread = threading.Thread(target=lambda: rclpy.spin(controller))
            ros_thread.daemon = True
            ros_thread.start()

            # 主线程处理手动控制
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