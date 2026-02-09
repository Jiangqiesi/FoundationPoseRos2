#!/bin/bash

cd ~/IML/FoundationPoseROS2/rm_moveit_config

echo "正在 source ROS2 环境..."
source /opt/ros/humble/setup.bash

if [ ! -d "install" ]; then
    echo "正在编译工作空间..."
    colcon build --symlink-install --packages-select rm_75_6f_description rm_moveit2
    if [ $? -ne 0 ]; then
        echo "编译失败!"
        exit 1
    fi
fi

echo "正在 source 工作空间..."
source install/setup.bash

echo "启动 MoveIt2 Demo (后台运行)..."
ros2 launch rm_moveit2 demo.launch.py & DEMO_PID=$!

echo "等待 MoveIt2 启动..."
sleep 5

# 运行桥接节点
echo "启动测试脚本..."
cd ~/IML/FoundationPoseROS2
python3 /home/ym/IML/FoundationPoseROS2/simple_moveit_controller.py "$@"

# 退出时清理
echo "正在关闭 Demo..."
kill $DEMO_PID 2>/dev/null || true
wait $DEMO_PID 2>/dev/null || true
