#!/bin/bash

cd ~/IML/FoundationPoseROS2/rm_moveit_config

echo "正在source ROS2环境..."
source /opt/ros/humble/setup.bash

if [ ! -d "install" ]; then
    echo "检测到未编译,开始编译MoveIt2配置包..."
    colcon build --symlink-install
    if [ $? -ne 0 ]; then
        echo "编译失败!"
        exit 1
    fi
    echo "编译完成!"
fi

echo "正在source工作空间..."
source install/setup.bash

echo "启动 MoveIt2 + FoundationPose 控制器..."
ros2 launch ~/IML/FoundationPoseROS2/foundationpose_controller.launch.py robot_ip:=192.168.0.17 object_ids:=1