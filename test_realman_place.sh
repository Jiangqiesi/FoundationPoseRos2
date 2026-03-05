#!/bin/bash
# filepath: /home/ym/IML/FoundationPoseROS2/test_kuka_moveit.sh

cd ~/gazebo_grasp/gazebo_ws

echo "正在 source ROS2 环境..."
source /opt/ros/humble/setup.bash

echo "正在 source 工作空间..."
source install/setup.bash

echo "启动 MoveIt2 Demo (后台运行)..."
ros2 launch robot_moveit_config demo.launch.py &
DEMO_PID=$!

echo "等待 MoveIt2 启动..."
sleep 5

echo "启动测试脚本..."
cd ~/gazebo_grasp/FoundationPoseROS2
python3 foundationpose_realman_place.py --robot-ip 192.168.0.17 --objects 1 2 3 4 5

# 清理
echo "正在关闭 Demo..."
kill $DEMO_PID
wait $DEMO_PID 2>/dev/null