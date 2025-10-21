#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
import os


def generate_launch_description():
    # 声明参数
    robot_ip = LaunchConfiguration('robot_ip', default='192.168.0.17')
    object_ids = LaunchConfiguration('object_ids', default='1')
    
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value='192.168.0.17')
    object_ids_arg = DeclareLaunchArgument('object_ids', default_value='1')

    # 基础路径
    base_path = "/home/ym/IML/FoundationPoseROS2/rm_moveit_config/src"
    script_path = "/home/ym/IML/FoundationPoseROS2/foundationpose_moveit2_controller.py"
    
    # 构建 MoveIt 配置
    try:
        moveit_config = (
            MoveItConfigsBuilder("rm_75_6f_description", package_name="rm_moveit2")
            .robot_description(file_path=f"{base_path}/rm_75_6f_description/urdf/rm_75_6f_description.urdf")
            .robot_description_semantic(file_path=f"{base_path}/rm_moveit2/config/rm_75_6f_description.srdf")
            .trajectory_execution(file_path=f"{base_path}/rm_moveit2/config/moveit_controllers.yaml")
            .planning_pipelines(pipelines=["ompl"])
            .to_moveit_configs()
        )
    except Exception as e:
        print(f"警告: MoveIt配置构建失败: {e}")
        print("将使用直接SDK控制模式")

    # 使用 ExecuteProcess 直接运行 Python 脚本
    controller_process = ExecuteProcess(
        cmd=[
            'python3',
            script_path,
            '--objects', object_ids,
            '--robot-ip', robot_ip,
        ],
        output='screen',
        shell=False,
    )

    return LaunchDescription([
        robot_ip_arg,
        object_ids_arg,
        controller_process,
    ])