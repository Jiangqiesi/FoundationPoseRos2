#!/bin/bash
# start_servers.sh - 一键启动所有服务端进程（MoveIt2 + 感知服务 + 控制服务）
# 用法: bash start_servers.sh --robot-ip 192.168.0.17 [--objects "1 2 3 4 5"]

# ============================================================================
# 参数解析
# ============================================================================
ROBOT_IP=""
OBJECTS="1 2 3 4 5"

while [[ $# -gt 0 ]]; do
  case $1 in
    --robot-ip)
      ROBOT_IP="$2"
      shift 2
      ;;
    --objects)
      OBJECTS="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: bash start_servers.sh --robot-ip <IP> [--objects \"1 2 3 4 5\"]"
      exit 1
      ;;
  esac
done

# 验证必需参数
if [ -z "$ROBOT_IP" ]; then
  echo "错误: --robot-ip 参数必需"
  echo "用法: bash start_servers.sh --robot-ip <IP> [--objects \"1 2 3 4 5\"]"
  exit 1
fi

echo "=========================================="
echo "启动所有服务端进程"
echo "=========================================="
echo "机械臂 IP: $ROBOT_IP"
echo "物体 ID: $OBJECTS"
echo ""

# ============================================================================
# 初始化 ROS2 环境
# ============================================================================
echo "[1/6] Source ROS2 环境..."
source /opt/ros/humble/setup.bash
if [ $? -ne 0 ]; then
  echo "错误: 无法 source ROS2 环境"
  exit 1
fi

# ============================================================================
# 启动 MoveIt2
# ============================================================================
echo "[2/6] 启动 MoveIt2 Demo..."
cd ~/IML/FoundationPoseROS2-client-server-work/rm_moveit_config
if [ ! -f "install/setup.bash" ]; then
  echo "错误: MoveIt2 工作空间未编译，请先运行 colcon build"
  exit 1
fi
source install/setup.bash

ros2 launch rm_moveit2 demo.launch.py &
DEMO_PID=$!
echo "MoveIt2 PID: $DEMO_PID"

echo "[3/6] 等待 MoveIt2 就绪..."
sleep 5

# ============================================================================
# Source foundationpose_msgs（如果存在）
# ============================================================================
echo "[4/6] Source foundationpose_msgs（如果需要）..."
if [ -d "~/IML/FoundationPoseROS2-client-server-work/foundationpose_msgs/install" ]; then
  cd ~/IML/FoundationPoseROS2-client-server-work/foundationpose_msgs
  if [ -f "install/setup.bash" ]; then
    source install/setup.bash
  fi
fi

# ============================================================================
# 启动感知服务器
# ============================================================================
echo "[5/6] 启动感知服务器..."
cd ~/IML/FoundationPoseROS2-client-server-work
python foundationpose_perception_server.py &
PERCEPTION_PID=$!
echo "感知服务器 PID: $PERCEPTION_PID"

# ============================================================================
# 启动控制服务器
# ============================================================================
echo "[6/6] 启动控制服务器..."
python foundationpose_realman_server.py --robot-ip "$ROBOT_IP" --objects $OBJECTS &
CONTROL_PID=$!
echo "控制服务器 PID: $CONTROL_PID"

# ============================================================================
# 打印状态和客户端命令示例
# ============================================================================
sleep 2
echo ""
echo "=========================================="
echo "✓ 所有服务端已启动"
echo "=========================================="
echo ""
echo "MoveIt2 Demo      PID: $DEMO_PID"
echo "感知服务器       PID: $PERCEPTION_PID"
echo "控制服务器       PID: $CONTROL_PID"
echo ""
echo "客户端命令示例:"
echo "  python foundationpose_cli.py status"
echo "  python foundationpose_cli.py pick --objects 1 2 --target 5"
echo "  python foundationpose_cli.py gripper open"
echo "  python foundationpose_cli.py resegment --force"
echo ""
echo "按 Ctrl+C 优雅退出所有进程..."
echo "=========================================="
echo ""

# ============================================================================
# 信号处理（SIGINT/SIGTERM）
# ============================================================================
cleanup() {
  echo ""
  echo "正在关闭所有服务..."
  
  if [ ! -z "$CONTROL_PID" ] && kill -0 $CONTROL_PID 2>/dev/null; then
    echo "关闭控制服务器 (PID: $CONTROL_PID)..."
    kill $CONTROL_PID
  fi
  
  if [ ! -z "$PERCEPTION_PID" ] && kill -0 $PERCEPTION_PID 2>/dev/null; then
    echo "关闭感知服务器 (PID: $PERCEPTION_PID)..."
    kill $PERCEPTION_PID
  fi
  
  if [ ! -z "$DEMO_PID" ] && kill -0 $DEMO_PID 2>/dev/null; then
    echo "关闭 MoveIt2 Demo (PID: $DEMO_PID)..."
    kill $DEMO_PID
  fi
  
  # 等待所有后台进程退出
  wait $DEMO_PID 2>/dev/null
  wait $PERCEPTION_PID 2>/dev/null
  wait $CONTROL_PID 2>/dev/null
  
  echo "✓ 所有服务已关闭"
  exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# 等待所有后台进程
# ============================================================================
wait
