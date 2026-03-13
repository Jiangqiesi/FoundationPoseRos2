#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoundationPose ROS2 命令行接口
基于 foundationpose_client.py 的 API 封装
"""

import sys
import argparse
import json
from foundationpose_client import FoundationPoseClient


def cmd_pick(args, client):
    """执行抓取放置任务"""

    def feedback_callback(feedback_msg):
        """打印 Action Feedback 进度"""
        fb = feedback_msg.feedback
        print(
            f"[{fb.current_stage}] 物体 {fb.current_object_id} "
            f"({fb.completed_count}/{fb.total_count} 完成): {fb.status_message}"
        )

    try:
        result = client.pick_place(
            object_ids=args.objects,
            target_id=args.target,
            enable_grasp=args.enable_grasp,
            offset_z=args.offset_z,
            approach_distance=args.approach_distance,
            lift_height=args.lift_height,
            feedback_callback=feedback_callback,
        )

        print(f"\n结果: {result['message']}")
        if result["completed_objects"]:
            print(f"成功: {result['completed_objects']}")
        if result["failed_objects"]:
            print(f"失败: {result['failed_objects']}")

        return 0 if result["success"] else 1

    except KeyboardInterrupt:
        print("\n\n检测到 Ctrl+C，取消当前任务...")
        client.cancel_current_action()
        return 1


def cmd_gripper(args, client):
    """控制夹爪"""
    try:
        if args.action == "open":
            result = client.open_gripper()
        else:
            result = client.close_gripper()

        print(f"{result['message']}")
        return 0 if result["success"] else 1

    except Exception as e:
        print(f"夹爪控制失败: {e}")
        return 1


def cmd_resegment(args, client):
    """请求重新分割"""
    try:
        result = client.resegment(force=args.force)
        print(f"{result['message']}")
        return 0 if result["success"] else 1

    except Exception as e:
        print(f"重新分割失败: {e}")
        return 1


def cmd_assign(args, client):
    """分配网格模型到掩码"""
    try:
        if len(args.meshes) != len(args.masks):
            print("错误: --meshes 和 --masks 数量必须相同")
            return 1

        result = client.assign_models(mesh_paths=args.meshes, mask_indices=args.masks)
        print(f"{result['message']}")
        return 0 if result["success"] else 1

    except Exception as e:
        print(f"模型分配失败: {e}")
        return 1


def cmd_status(args, client):
    """查询系统状态"""
    try:
        status = client.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0

    except Exception as e:
        print(f"状态查询失败: {e}")
        return 1


def cmd_param(args, client):
    """更新运动参数"""
    try:
        result = client.update_param(param_name=args.name, value=args.value)
        print(f"{result['message']}")
        return 0 if result["success"] else 1

    except Exception as e:
        print(f"参数更新失败: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="FoundationPose ROS2 命令行工具")

    subparsers = parser.add_subparsers(dest="command", help="可用命令", required=True)

    # pick 子命令
    pick_parser = subparsers.add_parser("pick", help="执行抓取放置任务")
    pick_parser.add_argument(
        "--objects", nargs="+", type=int, required=True, help="要抓取的物体 ID 列表"
    )
    pick_parser.add_argument(
        "--target", type=int, default=0, help="放置目标 ID（默认: 0）"
    )
    pick_parser.add_argument(
        "--enable-grasp", type=bool, default=True, help="是否启用夹爪控制（默认: True）"
    )
    pick_parser.add_argument(
        "--offset-z", type=float, default=0.133, help="Z 轴放置偏移（默认: 0.133）"
    )
    pick_parser.add_argument(
        "--approach-distance",
        type=float,
        default=0.1,
        help="靠近物体时的距离（默认: 0.1）",
    )
    pick_parser.add_argument(
        "--lift-height", type=float, default=0.1, help="提升高度（默认: 0.1）"
    )

    # gripper 子命令
    gripper_parser = subparsers.add_parser("gripper", help="控制夹爪")
    gripper_parser.add_argument("action", choices=["open", "close"], help="夹爪动作")

    # resegment 子命令
    resegment_parser = subparsers.add_parser("resegment", help="请求重新分割物体")
    resegment_parser.add_argument("--force", action="store_true", help="强制重新分割")

    # assign 子命令
    assign_parser = subparsers.add_parser("assign", help="分配网格模型到掩码")
    assign_parser.add_argument(
        "--meshes", nargs="+", required=True, help="网格文件路径列表"
    )
    assign_parser.add_argument(
        "--masks", nargs="+", type=int, required=True, help="对应的掩码索引列表"
    )

    # status 子命令
    subparsers.add_parser("status", help="查询系统状态")

    # param 子命令
    param_parser = subparsers.add_parser("param", help="更新运动参数")
    param_parser.add_argument("name", type=str, help="参数名称")
    param_parser.add_argument("value", type=float, help="参数值")

    args = parser.parse_args()

    # 创建客户端并执行命令
    try:
        with FoundationPoseClient() as client:
            print("等待服务端就绪...")
            client.wait_for_servers(timeout_sec=30.0)
            print("所有服务端已就绪\n")

            # 根据子命令分发
            if args.command == "pick":
                exit_code = cmd_pick(args, client)
            elif args.command == "gripper":
                exit_code = cmd_gripper(args, client)
            elif args.command == "resegment":
                exit_code = cmd_resegment(args, client)
            elif args.command == "assign":
                exit_code = cmd_assign(args, client)
            elif args.command == "status":
                exit_code = cmd_status(args, client)
            elif args.command == "param":
                exit_code = cmd_param(args, client)
            else:
                print(f"未知命令: {args.command}")
                exit_code = 1

            sys.exit(exit_code)

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
