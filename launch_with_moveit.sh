#!/bin/bash

cd ~/IML/FoundationPoseROS2

echo "激活 Conda 环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate foundationpose_ros

echo "Source ROS2 环境..."
source /opt/ros/humble/setup.bash

# 如果有 MoveIt 工作空间,source 它
if [ -d "rm_moveit_config/install" ]; then
    echo "Source MoveIt 工作空间..."
    source rm_moveit_config/install/setup.bash
fi
source ~/ws_moveit2_src/install/setup.bash

echo "启动 FoundationPose 控制器..."
ros2 launch foundationpose_simple.launch.py robot_ip:=192.168.0.17 object_ids:=1

echo "控制器已关闭"