#!/bin/bash

# FoundationPose-Realman机械臂控制脚本
# 用法示例:
#   ./run_foundationpose_realman.sh                           # 默认模式：手动控制物体1，启用抓取
#   ./run_foundationpose_realman.sh --auto-move               # 自动模式：收到位姿后立即移动并抓取
#   ./run_foundationpose_realman.sh --objects 1 2 3           # 监控多个物体
#   ./run_foundationpose_realman.sh --robot-ip 192.168.0.18   # 指定机械臂IP
#   ./run_foundationpose_realman.sh --disable-grasp           # 禁用抓取功能，仅移动
#   ./run_foundationpose_realman.sh --lift-height 0.15        # 设置抓取后提升高度为150mm

echo "启动FoundationPose-Realman集成控制器..."

# 激活ROS2环境（如果需要）
# source /opt/ros/humble/setup.bash
# source ~/your_workspace/install/setup.bash

# 运行Python脚本
python3 foundationpose_realman_controller.py "$@"