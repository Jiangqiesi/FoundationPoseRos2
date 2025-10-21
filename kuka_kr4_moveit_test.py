#!/usr/bin/env python3
"""
KUKA KR4 MoveIt2 Python 测试脚本
功能：
1. 初始化 MoveItPy
2. 移动到预定义位置
3. 笛卡尔路径规划
4. 碰撞检测测试
"""

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)
from moveit.core.robot_state import RobotState
from geometry_msgs.msg import PoseStamped
from moveit_configs_utils import MoveItConfigsBuilder
import time
import sys


class KukaKR4MoveItTest(Node):
    def __init__(self):
        super().__init__('kuka_kr4_moveit_test')
        
        self.get_logger().info('='*60)
        self.get_logger().info('KUKA KR4 MoveIt2 Python 测试程序')
        self.get_logger().info('='*60)
        
        try:
            # 初始化 MoveIt2
            self.initialize_moveit()
            
            # 获取机械臂信息
            self.print_robot_info()
            
        except Exception as e:
            self.get_logger().error(f'初始化失败: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            sys.exit(1)
    
    def initialize_moveit(self):
        """初始化 MoveIt2 配置"""
        self.get_logger().info('正在初始化 MoveIt2...')
        
        # 构建 MoveIt 配置
        moveit_config = (
            MoveItConfigsBuilder(
                robot_name="kuka_kr4",
                package_name="kuka_kr4_moveit2"
            )
            .robot_description(
                file_path="config/kuka_kr4_description.urdf.xacro"
            )
            .trajectory_execution(
                file_path="config/moveit_controllers.yaml"
            )
            .moveit_cpp(
                file_path="config/motion_planning_python_api_tutorial.yaml"
            )
            .to_moveit_configs()
        )
        
        # moveit_py_node = Node(
        #     name="moveit_py",
        #     package="moveit2_tutorials",
        #     executable=LaunchConfiguration("example_file"),
        #     output="both",
        #     parameters=[moveit_config.to_dict()],
        # )
        # 初始化 MoveItPy
        self.moveit = MoveItPy(
            node_name="moveit_py",
            config_dict=moveit_config.to_dict()
        )
        
        # 获取规划组件
        self.arm = self.moveit.get_planning_component("kuka_kr4_group")
        self.robot_model = self.moveit.get_robot_model()
        self.robot_state = self.moveit.get_planning_scene_monitor()
        
        self.get_logger().info('✓ MoveIt2 初始化成功')
    
    def print_robot_info(self):
        """打印机器人信息"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('机器人信息:')
        self.get_logger().info('='*60)
        
        # 获取关节组
        joint_model_group = self.robot_model.get_joint_model_group("kuka_kr4_group")
        
        # # 打印关节名称
        # joint_names = joint_model_group.get_active_joint_model_names()
        # self.get_logger().info(f'活动关节数量: {len(joint_names)}')
        # self.get_logger().info(f'关节名称: {joint_names}')
        
        # # 获取当前关节状态
        # current_state = self.robot_state
        # current_positions = []
        # for joint_name in joint_names:
        #     pos = current_state.get_joint_positions(joint_name)
        #     current_positions.append(pos[0] if len(pos) > 0 else 0.0)
        
        # self.get_logger().info(f'当前关节位置 (弧度): {[f"{p:.3f}" for p in current_positions]}')
        
        # 打印末端执行器信息
        ee_link = joint_model_group.get_link_model_names()[-1]
        self.get_logger().info(f'末端执行器链接: {ee_link}')
        
        self.get_logger().info('='*60 + '\n')
    
    def move_to_joint_values(self, joint_values, pipeline="ompl"):
        """
        移动到指定关节角度
        
        Args:
            joint_values: 关节角度列表 (弧度)
            pipeline: 规划管道 ("ompl" 或 "pilz_industrial_motion_planner")
        """
        self.get_logger().info(f'\n规划移动到关节位置: {[f"{v:.3f}" for v in joint_values]}')
        self.get_logger().info(f'使用规划器: {pipeline}')
        
        try:
            # 设置目标关节状态
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(configuration_name="")
            
            robot_state = RobotState(self.robot_model)
            joint_model_group = self.robot_model.get_joint_model_group("kuka_kr4_group")
            joint_names = joint_model_group.get_active_joint_model_names()
            
            for i, joint_name in enumerate(joint_names):
                if i < len(joint_values):
                    robot_state.set_joint_positions(joint_name, [joint_values[i]])
            
            self.arm.set_goal_state(robot_state=robot_state)
            
            # 规划
            self.get_logger().info('正在规划...')
            plan_result = self.arm.plan()
            
            if plan_result:
                self.get_logger().info('✓ 规划成功!')
                
                # 执行
                self.get_logger().info('正在执行运动...')
                self.robot_state = self.moveit.get_planning_scene_monitor().current_state
                self.moveit.execute(plan_result.trajectory, controllers=[])
                
                self.get_logger().info('✓ 执行完成')
                time.sleep(1)
                return True
            else:
                self.get_logger().error('✗ 规划失败')
                return False
                
        except Exception as e:
            self.get_logger().error(f'运动失败: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def move_to_pose(self, x, y, z, roll, pitch, yaw, pipeline="ompl"):
        """
        移动到指定笛卡尔位姿
        
        Args:
            x, y, z: 位置 (米)
            roll, pitch, yaw: 姿态 (弧度)
            pipeline: 规划管道
        """
        from scipy.spatial.transform import Rotation as R
        
        self.get_logger().info(f'\n规划移动到笛卡尔位姿:')
        self.get_logger().info(f'  位置: [{x:.3f}, {y:.3f}, {z:.3f}]')
        self.get_logger().info(f'  姿态: [{roll:.3f}, {pitch:.3f}, {yaw:.3f}]')
        self.get_logger().info(f'使用规划器: {pipeline}')
        
        try:
            # 创建目标位姿
            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "world"
            pose_goal.pose.position.x = x
            pose_goal.pose.position.y = y
            pose_goal.pose.position.z = z
            
            # 转换欧拉角到四元数
            quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
            pose_goal.pose.orientation.x = quat[0]
            pose_goal.pose.orientation.y = quat[1]
            pose_goal.pose.orientation.z = quat[2]
            pose_goal.pose.orientation.w = quat[3]
            
            # 设置目标
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="link_6")
            
            # 规划
            self.get_logger().info('正在规划...')
            plan_result = self.arm.plan()
            
            if plan_result:
                self.get_logger().info('✓ 规划成功!')
                
                # 执行
                self.get_logger().info('正在执行运动...')
                self.moveit.execute(plan_result.trajectory, controllers=[])
                
                self.get_logger().info('✓ 执行完成')
                time.sleep(1)
                return True
            else:
                self.get_logger().error('✗ 规划失败')
                return False
                
        except Exception as e:
            self.get_logger().error(f'运动失败: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def move_to_named_target(self, target_name):
        """
        移动到命名配置
        
        Args:
            target_name: 配置名称 (如 "home", "ready" 等)
        """
        self.get_logger().info(f'\n移动到命名配置: {target_name}')
        
        try:
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(configuration_name=target_name)
            
            plan_result = self.arm.plan()
            
            if plan_result:
                self.get_logger().info('✓ 规划成功!')
                self.moveit.execute(plan_result.trajectory, controllers=[])
                self.get_logger().info('✓ 执行完成')
                time.sleep(1)
                return True
            else:
                self.get_logger().error('✗ 规划失败')
                return False
                
        except Exception as e:
            self.get_logger().error(f'运动失败: {e}')
            return False
    
    def get_available_named_targets(self):
        """获取所有可用的命名配置"""
        joint_model_group = self.robot_model.get_joint_model_group("kuka_kr4_group")
        # 注意：这个API可能因MoveIt2版本而异
        try:
            # 尝试从SRDF获取命名状态
            return ["home", "ready"]  # 需要根据你的SRDF文件调整
        except:
            return []
    
    def run_interactive_test(self):
        """交互式测试模式"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('交互式测试模式')
        self.get_logger().info('='*60)
        
        while rclpy.ok():
            print("\n可用命令:")
            print("1. 移动到零位")
            print("2. 移动到自定义关节角度")
            print("3. 移动到笛卡尔位姿")
            print("4. 显示当前状态")
            print("5. 测试序列运动")
            print("q. 退出")
            
            choice = input("\n请选择: ").strip()
            
            if choice == '1':
                self.move_to_joint_values([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            elif choice == '2':
                try:
                    joints_str = input("输入6个关节角度 (弧度，空格分隔): ")
                    joint_values = [float(x) for x in joints_str.split()]
                    if len(joint_values) == 6:
                        self.move_to_joint_values(joint_values)
                    else:
                        print("错误：需要6个关节角度")
                except ValueError:
                    print("错误：输入格式不正确")
            
            elif choice == '3':
                try:
                    x = float(input("X (米): "))
                    y = float(input("Y (米): "))
                    z = float(input("Z (米): "))
                    roll = float(input("Roll (弧度): "))
                    pitch = float(input("Pitch (弧度): "))
                    yaw = float(input("Yaw (弧度): "))
                    self.move_to_pose(x, y, z, roll, pitch, yaw)
                except ValueError:
                    print("错误：输入格式不正确")
            
            elif choice == '4':
                self.print_robot_info()
            
            elif choice == '5':
                self.run_test_sequence()
            
            elif choice.lower() == 'q':
                print("退出程序...")
                break
            
            else:
                print("无效选择")
    
    def run_test_sequence(self):
        """运行测试序列"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('开始测试序列')
        self.get_logger().info('='*60)
        
        # 测试1: 移动到零位
        self.get_logger().info('\n[测试 1/3] 移动到零位')
        self.move_to_joint_values([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(2)
        
        # 测试2: 移动到测试位置1
        self.get_logger().info('\n[测试 2/3] 移动到测试位置1')
        self.move_to_joint_values([0.5, -0.5, 0.5, 0.0, 0.5, 0.0])
        time.sleep(2)
        
        # 测试3: 移动到测试位置2
        self.get_logger().info('\n[测试 3/3] 移动到测试位置2')
        self.move_to_joint_values([-0.5, 0.5, -0.5, 0.0, -0.5, 0.0])
        time.sleep(2)
        
        # 返回零位
        self.get_logger().info('\n返回零位')
        self.move_to_joint_values([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('✓ 测试序列完成')
        self.get_logger().info('='*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='KUKA KR4 MoveIt2 测试脚本')
    parser.add_argument('--interactive', action='store_true',
                       help='启动交互式测试模式')
    parser.add_argument('--test-sequence', action='store_true',
                       help='运行自动测试序列')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        node = KukaKR4MoveItTest()
        
        if args.interactive:
            node.run_interactive_test()
        elif args.test_sequence:
            node.run_test_sequence()
        else:
            # 默认：显示信息后进入交互模式
            print("\n提示：使用 --interactive 进入交互模式，或 --test-sequence 运行测试序列")
            node.run_interactive_test()
        
    except KeyboardInterrupt:
        print("\n收到退出信号")
    except Exception as e:
        print(f"程序错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()