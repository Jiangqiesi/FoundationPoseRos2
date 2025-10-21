#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import os


def generate_launch_description():
    # 声明启动参数
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.0.17',
        description='机械臂IP地址'
    )
    
    object_ids_arg = DeclareLaunchArgument(
        'object_ids',
        default_value='1',
        description='要跟踪的物体ID (多个用空格分隔)'
    )
    
    auto_move_arg = DeclareLaunchArgument(
        'auto_move',
        default_value='false',
        description='是否启用自动移动模式'
    )
    
    enable_grasp_arg = DeclareLaunchArgument(
        'enable_grasp',
        default_value='true',
        description='是否启用抓取功能'
    )

    # 基础路径
    base_path = "/home/ym/IML/FoundationPoseROS2/rm_moveit_config/src"
    
    # 构建 MoveIt 配置
    moveit_config = (
        MoveItConfigsBuilder("rm_75_6f_description", package_name="rm_moveit2")
        .robot_description(file_path=f"{base_path}/rm_75_6f_description/urdf/rm_75_6f_description.urdf")
        .robot_description_semantic(file_path=f"{base_path}/rm_moveit2/config/rm_75_6f_description.srdf")
        .trajectory_execution(file_path=f"{base_path}/rm_moveit2/config/moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl"],
            default_planning_pipeline="ompl"
        )
        .to_moveit_configs()
    )

    # 获取配置字典
    moveit_params = moveit_config.to_dict()
    
    # 手动添加 OMPL 配置 (如果配置文件不完整)
    if 'planning_pipelines' not in moveit_params:
        moveit_params['planning_pipelines'] = {}
    
    # 添加 OMPL 规划器配置
    moveit_params['planning_pipelines']['ompl'] = {
        'planning_plugin': 'ompl_interface/OMPLPlanner',
        'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/ResolveConstraintFrames default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints',
        'start_state_max_bounds_error': 0.1,
    }

    # 控制器节点
    controller_node = Node(
        package='your_package_name',  # 暂时用 package 名,如果没有包就用 executable 的完整路径
        executable='foundationpose_moveit2_controller.py',
        name='foundationpose_moveit2_controller',
        output='screen',
        parameters=[
            moveit_params,
            {
                'use_sim_time': False,
            }
        ],
        arguments=[
            '--objects', LaunchConfiguration('object_ids'),
            '--robot-ip', LaunchConfiguration('robot_ip'),
        ]
    )

    return LaunchDescription([
        robot_ip_arg,
        object_ids_arg,
        auto_move_arg,
        enable_grasp_arg,
        controller_node,
    ])