import numpy as np
from scipy.spatial.transform import Rotation as R

def transform_grasp_pose(obj_pos, obj_quat, grasp_pos_local, grasp_quat_local):
    """
    将物体坐标系下的抓取位姿转换到世界坐标系下。
    
    参数:
    obj_pos: [x, y, z] 物体在世界坐标系的位置
    obj_quat: [x, y, z, w] 物体在世界坐标系的旋转四元数
    grasp_pos_local: [x, y, z] 抓取点相对于物体的中心的位置
    grasp_quat_local: [x, y, z, w] 抓取点相对于物体的旋转
    
    返回:
    final_pos: [x, y, z] 抓取点在世界坐标系的位置
    final_quat: [x, y, z, w] 抓取点在世界坐标系的旋转
    """
    
    # 1. 构建物体(Object)到世界(World)的变换矩阵 T_world_obj
    r_obj = R.from_quat(obj_quat)
    T_world_obj = np.eye(4)
    T_world_obj[:3, :3] = r_obj.as_matrix()
    T_world_obj[:3, 3] = obj_pos

    # 2. 构建抓取点(Grasp)到物体(Object)的变换矩阵 T_obj_grasp
    r_grasp = R.from_quat(grasp_quat_local)
    T_obj_grasp = np.eye(4)
    T_obj_grasp[:3, :3] = r_grasp.as_matrix()
    T_obj_grasp[:3, 3] = grasp_pos_local

    # 3. 矩阵相乘： T_world_grasp = T_world_obj * T_obj_grasp
    # 注意顺序：父坐标系 @ 子坐标系
    T_world_grasp = T_world_obj @ T_obj_grasp

    # 4. 提取结果
    final_pos = T_world_grasp[:3, 3]
    final_rot = R.from_matrix(T_world_grasp[:3, :3])
    final_quat = final_rot.as_quat()

    return final_pos, final_quat

# --- 测试数据 ---
# 假设物体位于 (1, 0, 0)，并且绕Z轴旋转了90度
object_position = [1.0, 0.0, 0.0]
object_quaternion = [0.0, 0.0, 0.70710678, 0.70710678] # z-axis 90 deg

# 假设抓取位姿在物体右侧 0.1m 处，且方向与物体一致
grasp_position_local = [0.1, 0.0, 0.0]
grasp_quaternion_local = [0.0, 0.0, 0.0, 1.0] # Identity (无额外旋转)

final_p, final_q = transform_grasp_pose(object_position, object_quaternion, 
                                  grasp_position_local, grasp_quaternion_local)

print("最终抓取位置 (World):", final_p)
print("最终抓取四元数 (World):", final_q)