#!/bin/bash
# filepath: /home/ym/IML/FoundationPoseROS2/test_kuka_moveit.sh

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

source ~/ws_moveit2_src/install/setup.bash

echo "启动 MoveIt2 Demo (后台运行)..."
ros2 launch rm_moveit2 demo.launch.py &
DEMO_PID=$!

echo "等待 MoveIt2 启动..."
sleep 5

echo "启动测试脚本..."
cd ~/IML/FoundationPoseROS2
python3 foundationpose_moveit2_controller.py \
        --objects 1 2 \
        --grasp-file demo_data/ship/ship3_test.yml \
        --place-pose 0.031 -0.28 0.304 3.14 0 -1.57 \
        --home-pose 0.30 -0.12 0.37 3.14 0 0 \
        --place-approach-offset 0.08 "$@" \
        --grasp-files 1:demo_data/cup/cup_test.yml 2:demo_data/ship/ship3_test.yml

# 清理
echo "正在关闭 Demo..."
kill $DEMO_PID
wait $DEMO_PID 2>/dev/null