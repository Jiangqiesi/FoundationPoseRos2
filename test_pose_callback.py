import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import sys

class PoseTestNode(Node):
    def __init__(self, object_ids):
        super().__init__('pose_test_node')
        
        self.object_ids = object_ids
        self.subscribers = {}
        self.latest_poses = {}
        
        # 为每个物体创建订阅者
        for obj_id in self.object_ids:
            topic_name = f'/Current_OBJ_position_{obj_id}'
            self.subscribers[obj_id] = self.create_subscription(
                PoseStamped, 
                topic_name, 
                lambda msg, id=obj_id: self.pose_callback(msg, id), 
                10
            )
            self.get_logger().info(f'订阅物体 {obj_id} 的位姿话题: {topic_name}')
        
        # 创建定时器，定期显示所有物体位姿
        self.timer = self.create_timer(5.0, self.display_poses)

    def pose_callback(self, msg, object_id):
        # 存储最新位姿
        self.latest_poses[object_id] = msg
        
        position = msg.pose.position
        orientation = msg.pose.orientation
        
        # 转换四元数到欧拉角 (roll, pitch, yaw)
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        
        # self.get_logger().info(f'物体 {object_id} 位姿更新:')
        # self.get_logger().info(f'  位置 (m): x={position.x:.4f}, y={position.y:.4f}, z={position.z:.4f}')
        # self.get_logger().info(f'  方向 (四元数): w={orientation.w:.4f}, x={orientation.x:.4f}, y={orientation.y:.4f}, z={orientation.z:.4f}')
        # self.get_logger().info(f'  方向 (欧拉角°): roll={euler[0]:.2f}, pitch={euler[1]:.2f}, yaw={euler[2]:.2f}')
        # self.get_logger().info(f'  时间戳: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        # self.get_logger().info('---')

    def display_poses(self):
        if not self.latest_poses:
            return
        
        print("\n" + "="*80)
        print("当前所有物体位姿汇总:")
        print("="*80)
        
        for obj_id in sorted(self.latest_poses.keys()):
            msg = self.latest_poses[obj_id]
            pos = msg.pose.position
            orient = msg.pose.orientation
            
            # 转换到欧拉角
            quat = [orient.x, orient.y, orient.z, orient.w]
            euler = R.from_quat(quat).as_euler('xyz', degrees=True)
            
            print(f"物体 {obj_id}:")
            print(f"  位置 (m): [{pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}]")
            print(f"  四元数: [{orient.w:.4f}, {orient.x:.4f}, {orient.y:.4f}, {orient.z:.4f}]")
            print(f"  欧拉角 (°): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
            print(f"  坐标系: {msg.header.frame_id}")
            print()

    def get_pose_for_grasping(self, object_id):
        """获取指定物体的位姿用于抓取规划"""
        if object_id not in self.latest_poses:
            self.get_logger().warn(f'未找到物体 {object_id} 的位姿信息')
            return None
        
        msg = self.latest_poses[object_id]
        pose_dict = {
            'position': {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z
            },
            'orientation': {
                'w': msg.pose.orientation.w,
                'x': msg.pose.orientation.x,
                'y': msg.pose.orientation.y,
                'z': msg.pose.orientation.z
            },
            'frame_id': msg.header.frame_id,
            'timestamp': msg.header.stamp
        }
        return pose_dict

def main():
    parser = argparse.ArgumentParser(description='测试FoundationPose发布的6D位姿')
    parser.add_argument('--objects', nargs='+', type=int, default=[1], 
                       help='要订阅的物体ID列表 (默认: [1])')
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        node = PoseTestNode(args.objects)
        print(f"开始订阅物体 {args.objects} 的位姿信息...")
        print("按 Ctrl+C 退出")
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n收到退出信号，正在关闭...")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()