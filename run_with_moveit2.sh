#!/bin/bash

cd ~/IML/FoundationPoseROS2

source /opt/ros/humble/setup.bash
source rm_moveit_config/install/setup.bash
source ~/ws_moveit2_src/install/setup.bash

echo "========================================"
echo "  FoundationPose + MoveIt2 控制器"
echo "========================================"
echo ""
echo "使用方法:"
echo "  python3 foundationpose_moveit2_controller.py --objects 1 --robot-ip 192.168.0.17"
echo ""
echo "注意: 请先在另一个终端运行:"
echo "  ./launch_moveit2.sh"
echo ""
echo "========================================"
echo ""

exec "$@"
