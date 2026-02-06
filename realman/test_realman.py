import time
from RealMan import RM_controller, initialize_robot
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e

def main():
    """
    主函数,用于测试机械臂的手动控制.
    """
    rm_controller = None
    try:
        # --- 初始化机械臂 ---
        # 请将 "192.168.0.18" 替换为您的机械臂的实际IP地址
        robot_ip = "192.168.0.17"
        print(f"正在连接到机械臂: {robot_ip}...")
        rm_controller = RM_controller(robot_ip, rm_thread_mode_e.RM_TRIPLE_MODE_E)
        
        # --- 获取并打印初始状态 ---
        initial_state = rm_controller.get_state()
        print(f"机械臂连接成功. 当前关节角度: {initial_state}")
        print("\n--- 机械臂手动控制测试 ---")
        print("请输入6D位姿，前3个元素为位置 (x, y, z)单位（m），后3个元素为姿态 (rx, ry, rz)单位（rad），以空格分隔.")
        print("例如: 0 0 0 0 0 0")
        print("输入 'q' 或 'quit' 退出程序.")

        # --- 控制循环 ---
        while True:
            user_input = input("\n请输入目标6D位姿 > ").strip()

            if user_input.lower() in ['q', 'quit']:
                print("程序退出.")
                break

            try:
                # --- 解析输入 ---
                target_pose_str = user_input.split()
                if len(target_pose_str) != 6:
                    print(f"错误: 需要输入6D位姿, 但您输入了 {len(target_pose_str)} 个.")
                    continue

                target_pose = [float(coord) for coord in target_pose_str]

                # --- 移动机械臂 ---
                print(f"正在移动到目标位姿: {target_pose}...")
                # 使用 movel (笛卡尔空间线性移动)
                success = rm_controller.movel(target_pose)
                
                if success == 0:
                    print("移动成功!")
                else:
                    print(f"移动失败, 错误码: {success}")

                time.sleep(1) # 等待一秒,让机械臂有时间完成移动
                current_state = rm_controller.get_state()
                print(f"移动后当前关节角度: {current_state}")

            except ValueError:
                print("错误: 输入无效. 请确保输入的是数字, 并以空格分隔.")
            except Exception as e:
                print(f"移动过程中发生错误: {e}")

    except ConnectionError as e:
        print(f"连接失败: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    finally:
        # --- 清理 ---
        if rm_controller:
            # 在程序结束时可以添加资源释放代码
            print("程序结束, 正在断开与机械臂的连接...")
            del rm_controller

if __name__ == "__main__":
    main()
