import rclpy
from geometry_msgs.msg import PoseStamped

def pose_callback(msg):
    # 获取物体位姿
    position = msg.pose.position  # x, y, z
    orientation = msg.pose.orientation  # w, x, y, z
    print(f"Position: [{position.x}, {position.y}, {position.z}]")
    print(f"Orientation: [{orientation.w}, {orientation.x}, {orientation.y}, {orientation.z}]")

# 订阅第一个物体的位姿
node.create_subscription(PoseStamped, '/Current_OBJ_position_1', pose_callback, 10)