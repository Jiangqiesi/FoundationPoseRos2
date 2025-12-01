#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化 STL 模型与坐标轴的小工具（基于 Open3D）

功能:
- 载入 STL 并显示三角网格
- 显示世界坐标轴 (X=红, Y=绿, Z=蓝)
- 可选: 将模型平移到几何中心
- 可选: 以欧拉角 (deg) 旋转模型 (XYZ 顺序)
- 可选: 调整坐标轴大小、网格线、背景色
- 支持保存截图（按 's' 键）

依赖:
    pip install open3d numpy

用法示例:
    python visualize_stl.py --stl path/to/model.stl --center --rotate 0,90,0 --axis-size 100
"""

import argparse
import numpy as np
import open3d as o3d

try:
    from open3d.visualization import rendering
except ImportError:  # pragma: no cover - graceful fallback if module missing
    rendering = None


def parse_args():
    p = argparse.ArgumentParser(description="Visualize STL with world axes (Open3D)")
    p.add_argument("--stl", required=True, help="STL 文件路径")
    p.add_argument("--center", action="store_true",
                   help="将模型平移到几何中心（质心对齐到原点）")
    p.add_argument("--rotate", default=None,
                   help="以度为单位的欧拉角，格式 'rx,ry,rz'，XYZ 顺序 (例如: 0,90,0)")
    p.add_argument("--axis-size", type=float, default=1.0,
                   help="坐标轴大小（默认 1.0）")
    p.add_argument("--mesh-color", default=None,
                   help="模型颜色，格式 'r,g,b'，范围[0,1]，如 '0.8,0.8,0.8'")
    p.add_argument("--wireframe", action="store_true",
                   help="启用线框（需渲染器支持）")
    p.add_argument("--bg", default=None,
                   help="背景颜色，格式 'r,g,b'，范围[0,1]，如 '1,1,1' 白色背景")
    p.add_argument("--save", default=None,
                   help="在无显示环境下保存离屏渲染到指定 PNG 路径")
    return p.parse_args()


def str_to_rgb(s):
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError("颜色应为 'r,g,b' 三个值")
    return np.clip(np.array(vals, dtype=float), 0.0, 1.0)


def main():
    args = parse_args()

    # 读取 STL
    mesh = o3d.io.read_triangle_mesh(args.stl)
    if mesh.is_empty():
        raise RuntimeError(f"读取失败或模型为空: {args.stl}")

    # 法线（用于正确渲染）
    mesh.compute_vertex_normals()

    # 可选：统一上色
    if args.mesh_color is not None:
        c = str_to_rgb(args.mesh_color)
        mesh.paint_uniform_color(c.tolist())

    # 可选：将模型移到几何中心（质心）
    if args.center:
        center = mesh.get_center()
        mesh.translate(-center)

    # 可选：欧拉角旋转 (XYZ 顺序)，单位：度
    if args.rotate:
        try:
            rx, ry, rz = [float(v) for v in args.rotate.split(",")]
        except Exception:
            raise ValueError("rotate 参数应为 'rx,ry,rz'，例如 '0,90,0'")
        R = mesh.get_rotation_matrix_from_xyz(np.deg2rad([rx, ry, rz]))
        mesh.rotate(R, center=(0.0, 0.0, 0.0))

    # 世界坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(args.axis_size), origin=[0, 0, 0]
    )

    # 设置渲染窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    window_created = vis.create_window(
        window_name="STL Viewer with Axes", width=1280, height=800, visible=True
    )

    def render_offscreen(output_path: str):
        if rendering is None:
            raise RuntimeError("当前 Open3D 版本不支持离屏渲染，请升级到 0.13+ 或提供可用的 DISPLAY。")

        width, height = 1280, 800
        renderer = rendering.OffscreenRenderer(width, height)
        scene = renderer.scene

        if args.wireframe:
            print("提示：离屏渲染暂不支持线框显示，已自动忽略 --wireframe 参数。")

        # 设置背景色
        if args.bg is not None:
            bg = str_to_rgb(args.bg)
            scene.set_background(np.array([bg[0], bg[1], bg[2], 1.0], dtype=np.float32))
        else:
            scene.set_background(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        # 添加几何体材质
        mesh_material = rendering.MaterialRecord()
        mesh_material.shader = "defaultLit"
        if args.mesh_color is not None:
            c = str_to_rgb(args.mesh_color)
            mesh_material.base_color = np.array([c[0], c[1], c[2], 1.0], dtype=np.float32)
        scene.add_geometry("mesh", mesh, mesh_material)

        axis_material = rendering.MaterialRecord()
        axis_material.shader = "defaultUnlit"
        scene.add_geometry("axis", axis, axis_material)

        # 设定相机位置（简单的斜上方视角）
        bounds = mesh.get_axis_aligned_bounding_box()
        bounds.extend(axis.get_axis_aligned_bounding_box())
        center = bounds.get_center()
        extent = bounds.get_extent()
        radius = np.linalg.norm(extent) if np.linalg.norm(extent) > 0 else 1.0
        cam_distance = radius * 2.5
        eye = center + np.array([cam_distance, cam_distance, cam_distance])
        scene.camera.look_at(center, eye, np.array([0.0, 0.0, 1.0]))
        scene.scene.set_lighting(rendering.Scene.LightingProfile.NO_SHADOWS, np.array([0.0, 0.0, 0.0]))

        image = renderer.render_to_image()
        o3d.io.write_image(output_path, image)
        print(f"已保存离屏渲染：{output_path}")

    if not window_created:
        vis.destroy_window()
        if args.save:
            render_offscreen(args.save)
            return
        raise RuntimeError("无法创建 Open3D 窗口，请确保 DISPLAY 可用，或使用 --save 进行离屏渲染。")

    # 设置背景色
    if args.bg is not None:
        bg = str_to_rgb(args.bg)
        opt = vis.get_render_option()
        opt.background_color = bg

    # 添加几何体
    vis.add_geometry(mesh)
    vis.add_geometry(axis)

    # 可选：线框显示（Open3D 的线框为叠加模式）
    if args.wireframe:
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.mesh_show_wireframe = True

    # 快捷键：保存截图
    def save_screenshot(vis_):
        vis_.capture_screen_image("screenshot.png", do_render=True)
        print("已保存截图：screenshot.png")
        return False

    vis.register_key_callback(ord("S"), save_screenshot)
    vis.register_key_callback(ord("s"), save_screenshot)

    # 相机视角优化
    view_control = vis.get_view_control()
    if view_control is None:
        vis.destroy_window()
        if args.save:
            render_offscreen(args.save)
            return
        raise RuntimeError(
            "Open3D 创建窗口失败（环境可能缺少 DISPLAY）。请在有显示的环境运行或添加 --save 进行离屏渲染。"
        )
    view_control.set_zoom(0.8)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
