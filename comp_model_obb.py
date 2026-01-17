import trimesh
import numpy as np

def analyze_model_obb(obj_path):
    """
    加载OBJ模型，计算OBB（定向包围盒），并返回其尺寸和旋转偏移量。
    """
    # 1. 加载模型
    try:
        mesh = trimesh.load(obj_path, force='mesh')
    except Exception as e:
        print(f"无法加载模型: {e}")
        return

    print(f"模型已加载: {obj_path}")
    print(f"顶点数: {len(mesh.vertices)}, 面数: {len(mesh.faces)}")

    # 2. 计算定向包围盒 (Oriented Bounding Box - OBB)
    # trimesh 使用凸包和惯性张量等方法估算最小体积的OBB
    obb = mesh.bounding_box_oriented

    # 3. 获取 OBB 的变换矩阵 (4x4)
    # 这个矩阵将一个中心在原点、轴对齐的单位立方体变换为当前的 OBB
    transform_matrix = obb.primitive.transform

    # 4. 提取旋转矩阵 (前3x3部分)
    # 修改点：添加 .copy() 以解决 "buffer source array is read-only" 错误
    rotation_matrix = transform_matrix[:3, :3].copy()

    # 5. 提取 OBB 的尺寸 (长宽高)
    # primitive.extents 返回的是未变换前的轴对齐尺寸
    extents = obb.primitive.extents

    # 6. 提取中心点位置
    obb_center = transform_matrix[:3, 3]

    # 7. 计算模型原始中心点（几种方式）
    # 方式1: 质心 (centroid) - 基于面积加权的中心
    model_centroid = mesh.centroid
    
    # 方式2: 顶点的几何中心 (bounding box center)
    model_bounds_center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    
    # 方式3: 重心 (center of mass) - 基于体积的质量中心
    model_center_mass = mesh.center_mass

    # 8. 计算偏差
    offset_from_centroid = obb_center - model_centroid
    offset_from_bounds = obb_center - model_bounds_center
    offset_from_mass = obb_center - model_center_mass

    print("-" * 30)
    print("计算结果:")
    print(f"OBB 尺寸 (长 x 宽 x 高): {extents}")
    print(f"\n中心点位置:")
    print(f"  模型质心 (centroid):        {model_centroid}")
    print(f"  模型边界框中心 (bounds):    {model_bounds_center}")
    print(f"  模型重心 (center of mass):  {model_center_mass}")
    print(f"  OBB 中心点:                 {obb_center}")
    
    print(f"\n中心点偏差 (OBB中心 - 模型中心):")
    print(f"  相对于质心的偏差:     {offset_from_centroid}")
    print(f"  相对于边界框中心偏差: {offset_from_bounds}")
    print(f"  相对于重心的偏差:     {offset_from_mass}")
    
    print(f"\n偏差距离:")
    print(f"  相对于质心:     {np.linalg.norm(offset_from_centroid):.6f}")
    print(f"  相对于边界框:   {np.linalg.norm(offset_from_bounds):.6f}")
    print(f"  相对于重心:     {np.linalg.norm(offset_from_mass):.6f}")

    print("\n旋转偏移量 (旋转矩阵 3x3):")
    print(rotation_matrix)
    
    # 如果需要欧拉角 (弧度)
    try:
        # 需要 scipy.spatial.transform
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=True)
        print(f"\n旋转偏移量 (欧拉角 XYZ, 度): {euler_angles}")
    except ImportError:
        print("\n提示: 安装 scipy 可查看欧拉角输出 (pip install scipy)")

    # 可视化 - 将模型和半透明OBB一起显示
    # 创建OBB的网格副本用于可视化
    obb_mesh = obb.to_mesh()
    
    # 设置OBB为半透明颜色 (RGBA: 绿色半透明)
    obb_mesh.visual.face_colors = [100, 200, 100, 80]
    
    # 设置原始模型的颜色
    mesh.visual.face_colors = [150, 150, 200, 255]  # 浅蓝色不透明
    
    # 创建场景并添加几何体
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name='model')
    scene.add_geometry(obb_mesh, geom_name='obb')
    
    # 创建中心点标记球体
    sphere_radius = np.min(extents) * 0.03  # 根据模型大小自适应球体半径
    
    # 模型质心 - 红色球体
    centroid_sphere = trimesh.creation.icosphere(radius=sphere_radius)
    centroid_sphere.apply_translation(model_centroid)
    centroid_sphere.visual.face_colors = [255, 0, 0, 255]  # 红色
    scene.add_geometry(centroid_sphere, geom_name='centroid')
    
    # 模型边界框中心 - 黄色球体
    bounds_sphere = trimesh.creation.icosphere(radius=sphere_radius)
    bounds_sphere.apply_translation(model_bounds_center)
    bounds_sphere.visual.face_colors = [255, 255, 0, 255]  # 黄色
    scene.add_geometry(bounds_sphere, geom_name='bounds_center')
    
    # 模型重心 - 橙色球体
    mass_sphere = trimesh.creation.icosphere(radius=sphere_radius)
    mass_sphere.apply_translation(model_center_mass)
    mass_sphere.visual.face_colors = [255, 165, 0, 255]  # 橙色
    scene.add_geometry(mass_sphere, geom_name='center_mass')
    
    # OBB中心 - 绿色球体
    obb_center_sphere = trimesh.creation.icosphere(radius=sphere_radius)
    obb_center_sphere.apply_translation(obb_center)
    obb_center_sphere.visual.face_colors = [0, 255, 0, 255]  # 绿色
    scene.add_geometry(obb_center_sphere, geom_name='obb_center')
    
    # 创建从各中心点到OBB中心的连接线
    # 质心到OBB中心 - 红色线
    line_centroid = trimesh.load_path(
        np.array([[model_centroid, obb_center]])
    )
    line_centroid.colors = [[255, 0, 0, 255]]
    scene.add_geometry(line_centroid, geom_name='line_centroid')
    
    # 边界框中心到OBB中心 - 黄色线
    line_bounds = trimesh.load_path(
        np.array([[model_bounds_center, obb_center]])
    )
    line_bounds.colors = [[255, 255, 0, 255]]
    scene.add_geometry(line_bounds, geom_name='line_bounds')
    
    # 重心到OBB中心 - 橙色线
    line_mass = trimesh.load_path(
        np.array([[model_center_mass, obb_center]])
    )
    line_mass.colors = [[255, 165, 0, 255]]
    scene.add_geometry(line_mass, geom_name='line_mass')
    
    print("\n可视化说明:")
    print("  红色球体/线: 模型质心 (centroid)")
    print("  黄色球体/线: 模型边界框中心 (bounds center)")
    print("  橙色球体/线: 模型重心 (center of mass)")
    print("  绿色球体:    OBB包围盒中心")
    print("  绿色半透明:  OBB包围盒")
    print("  蓝色:        原始模型")
    
    # 显示场景
    scene.show()

    return rotation_matrix, extents, {
        'obb_center': obb_center,
        'model_centroid': model_centroid,
        'model_bounds_center': model_bounds_center,
        'model_center_mass': model_center_mass,
        'offset_from_centroid': offset_from_centroid,
        'offset_from_bounds': offset_from_bounds,
        'offset_from_mass': offset_from_mass
    }

if __name__ == "__main__":
    # 替换为你的 obj 文件路径
    # 如果没有文件，trimesh 可以生成一个测试用的
    # mesh = trimesh.creation.box(extents=[2, 1, 0.5])
    # mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
    # mesh.export('test_model.obj')
    
    model_path = "demo_data/ship/ship3.obj" 
    
    # 为了演示，这里创建一个临时的测试文件
    test_mesh = trimesh.creation.box(extents=[10, 2, 1])
    # 随机旋转一下，模拟带有旋转偏移的模型
    random_rot = trimesh.transformations.rotation_matrix(np.pi/4, [0, 0, 1])
    test_mesh.apply_transform(random_rot)
    test_mesh.export('temp_test_model.obj')
    
    analyze_model_obb(model_path)