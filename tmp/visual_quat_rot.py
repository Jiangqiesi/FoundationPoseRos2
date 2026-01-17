import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def plot_rotated_axes(quaternion):
    """
    可视化四元数旋转。
    输入 quaternion: list or array [x, y, z, w]
    """
    # 1. 创建绘图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 2. 定义原始坐标轴基向量 (Identity)
    # X轴(红), Y轴(绿), Z轴(蓝)
    origin = np.array([0, 0, 0])
    basis = np.eye(3) 

    # 3. 计算旋转后的基向量
    # scipy 的 Rotation 默认输入格式为 [x, y, z, w]
    r = R.from_quat(quaternion)
    rotated_basis = r.apply(basis)

    # 4. 绘制辅助函数
    def draw_axes(ax, vectors, origin, style='-', alpha=1.0, label_prefix=''):
        colors = ['r', 'g', 'b'] # RGB 对应 XYZ
        labels = ['X', 'Y', 'Z']
        
        for i, vec in enumerate(vectors):
            ax.quiver(origin[0], origin[1], origin[2], 
                      vec[0], vec[1], vec[2], 
                      color=colors[i], linestyle=style, 
                      arrow_length_ratio=0.1, alpha=alpha, linewidth=2)
            # 添加标签
            ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, 
                    f"{label_prefix}{labels[i]}", color=colors[i])

    # 5. 绘制原始坐标系 (虚线，透明度低)
    draw_axes(ax, basis, origin, style='--', alpha=0.3, label_prefix='Orig_')

    # 6. 绘制旋转后的坐标系 (实线)
    draw_axes(ax, rotated_basis, origin, style='-', alpha=1.0, label_prefix='Rot_')

    # 7. 设置图形属性
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Quaternion Rotation: {quaternion}\n(Red=X, Green=Y, Blue=Z)')
    
    plt.show()

if __name__ == "__main__":
    # # 示例 1: 绕 Z 轴旋转 90 度
    # # 欧拉角转四元数参考: 0, 0, 90度 -> [0, 0, 0.707, 0.707]
    # q1 = [0, 0.7071068, 0, 0.7071068]
    # print(f"Visualizing q1: {q1}")
    # plot_rotated_axes(q1)

    # # 示例 2: 任意旋转
    # q2 = [0.1, 0.2, 0.3, 0.9] (未归一化，scipy会自动处理或你可以先归一化)
    q2 = [0, 1.0, 0, 0]  # 180度绕Y轴旋转
    # 归一化四元数
    q2 = q2 / np.linalg.norm(q2)
    print(f"Visualizing q2: {q2}")
    plot_rotated_axes(q2)