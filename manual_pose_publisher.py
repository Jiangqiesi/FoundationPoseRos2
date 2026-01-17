#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import sys

# transformation 坐标转换处理模块
# --- 选项 1: 使用 cam_2_base_transform 模块（进行相机坐标系 -> 机械臂坐标系转换） ---
# try:
#     from cam_2_base_transform import transformation
# except ImportError:
#     print("警告: 未找到 cam_2_base_transform 模块")
#     sys.exit(1)

# --- 选项 2: 直通模式 ---
def transformation(pose_array):
    # 输入: [x, y, z, qx, qy, qz, qw]
    # 输出: [x, y, z, x, y, z, w]
    return [pose_array[0], pose_array[1], pose_array[2], 
            pose_array[3], pose_array[4], pose_array[5], pose_array[6]]

class ManualPosePublisher(Node):
    def __init__(self, x, y, z, roll, pitch, yaw, frame_id, topic_name):
        """
        初始化ManualPosePublisher节点，用于发布手动设置的位姿信息
        
        Args:
            x (float): X轴坐标位置
            y (float): Y轴坐标位置
            z (float): Z轴坐标位置
            roll (float): 绕X轴旋转的欧拉角（单位：度）
            pitch (float): 绕Y轴旋转的欧拉角（单位：度）
            yaw (float): 绕Z轴旋转的欧拉角（单位：度）
            frame_id (str): 坐标系ID
            topic_name (str): 发布话题名称
        """
        super().__init__('manual_pose_publisher')
        
        self.publisher_ = self.create_publisher(PoseStamped, topic_name, 10)
        self.timer = self.create_timer(0.1, self.timer_callback) # 10Hz 发布频率
        self.frame_id = frame_id

        # 只打印一次的标志位
        self._printed_once = False

        # 目标位姿
        self.position = np.array([x, y, z]) # [x, y, z] 位置坐标
        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False) # 欧拉角 (弧度) => 旋转矩阵
        self.quaternion = r.as_quat() # [qx, qy, qz, qw] 四元数
        
        self.get_logger().info(f"节点已启动，正在发布手动位姿到话题: {topic_name}")
        self.get_logger().info(f"源位姿 (Pos): {self.position}")
        self.get_logger().info(f"源位姿 (RPY): {[roll, pitch, yaw]}")

    def timer_callback(self):
        # 1. 准备数据 [x, y, z, qx, qy, qz, qw]
        pose_array = np.concatenate((self.position, self.quaternion))

        # 2. 执行坐标转换 (模拟原代码中的 cam -> base，这里实际直通，未转换)
        try:
            transformed_pose = transformation(pose_array)
        except Exception as e:
            self.get_logger().error(f"坐标转换出错: {e}")
            return

        # 3. 创建一个PoseStamped消息对象，填充数据后通过publish()方法发布
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # 填充数据
        msg.pose.position.x = float(transformed_pose[0])
        msg.pose.position.y = float(transformed_pose[1])
        msg.pose.position.z = float(transformed_pose[2])
        
        msg.pose.orientation.x = float(transformed_pose[3])
        msg.pose.orientation.y = float(transformed_pose[4])
        msg.pose.orientation.z = float(transformed_pose[5])
        msg.pose.orientation.w = float(transformed_pose[6])

        # 4. 发布
        self.publisher_.publish(msg)
        
        # 打印调试信息
        if not self._printed_once:
            pos_str = f"Pos=[{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]"
            ori_str = f"Ori=[{msg.pose.orientation.x:.3f}, {msg.pose.orientation.y:.3f}, {msg.pose.orientation.z:.3f}, {msg.pose.orientation.w:.3f}]"
            self.get_logger().info(f"发布位姿: {pos_str}, {ori_str}")
            self._printed_once = True
def main():
    # 创建一个命令行参数解析器对象
    parser = argparse.ArgumentParser(description="手动发布 PoseStamped 消息用于测试")
    
    # 位置参数 (米)
    parser.add_argument('-x', type=float, default=0.0, help='X 坐标 (m)')
    parser.add_argument('-y', type=float, default=0.0, help='Y 坐标 (m)')
    parser.add_argument('-z', type=float, default=0.4, help='Z 坐标 (m)')
    
    # 姿态参数 (欧拉角，度)
    parser.add_argument('-R', '--roll', type=float, default=3.0, help='Roll (reg)')
    parser.add_argument('-P', '--pitch', type=float, default=0.0, help='Pitch (reg)')
    parser.add_argument('-Y', '--yaw', type=float, default=3.0, help='Yaw (reg)')
    
    # ROS 配置
    parser.add_argument('--topic', type=str, default='/Current_OBJ_position_1', help='发布的话题名称')
    parser.add_argument('--frame', type=str, default='base_link', help='Frame ID')

    args = parser.parse_args()

    rclpy.init()
    
    node = ManualPosePublisher(
        args.x, args.y, args.z,
        args.roll, args.pitch, args.yaw,
        args.frame, args.topic
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()