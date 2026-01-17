from Robotic_Arm.rm_robot_interface import *
import time
import numpy as np

class RM_controller:
    """
    RealMan机械臂控制器封装类。

    该类简化了与 `RoboticArm` 接口的交互，提供更高级的控制方法，
    如获取状态、移动和控制夹爪。
    """
    def __init__(self, rm_ip, thread_mode=None):
        """
        初始化RM_controller。

        Args:
            rm_ip (str): 机械臂的IP地址。
            thread_mode (rm_thread_mode_e, optional): 线程模式。
                例如 `rm_thread_mode_e.RM_TRIPLE_MODE_E` 用于三线程模式，
                可以提高通信的实时性和稳定性。默认为None（单线程）。
        
        Raises:
            ConnectionError: 如果无法连接或初始化机械臂。
        """
        try:
            # 调用底层接口初始化机器人
            result = initialize_robot(rm_ip, thread_mode)
            if result is None:
                raise ConnectionError(f"无法连接到IP为 {rm_ip} 的机械臂")
            self.arm_controller, self.handle = result
        except Exception as e:
            raise ConnectionError(f"初始化机械臂失败: {str(e)}")
        
        # 初始化内部状态变量
        self.is_controlling = False
        self.prev_tech_state = None
        self.arm_first_state = None
        self.gripper_close = False
        self.delta = [0, 0, 0, 0, 0, 0, 0]

    def get_state(self):
        """
        获取机械臂当前的关节角度状态。

        Returns:
            list: 包含7个关节角度的列表 (单位: 度)。
                  如果获取失败，可能返回空列表或引发异常。
        """
        # rm_get_current_arm_state 返回一个元组 (code, state_dict)
        # state_dict["joint"] 包含了关节角度列表
        return self.arm_controller.rm_get_current_arm_state()[1]["joint"]

    def get_gripper(self):
        """
        获取夹爪当前的位置。

        Returns:
            float: 夹爪的位置，范围从 0.0 (闭合) 到 1.0 (完全张开)。
        """
        # rm_get_gripper_state 返回 (code, gripper_dict)
        # gripper_dict["actpos"] 是实际位置，范围 0-1000
        return float(self.arm_controller.rm_get_gripper_state()[1]["actpos"]) / 1000.0

    def move_test(self):
        """用于测试与机械臂连接是否正常的函数。"""
        try:
            succ, arm_state = self.arm_controller.rm_get_current_arm_state()
            print("获取状态成功: ", succ == 0)
            print("当前状态: ", arm_state)
        except Exception as e:
            raise RuntimeError(f"测试移动时出错: {str(e)}")
        
    def movel(self, pose):
        """
        以线性方式移动机械臂到指定的笛卡尔空间位置。

        Args:
            pose (list): 包含6个元素的列表，表示目标位置和姿态。
                         前3个元素为位置 (x, y, z)，后3个元素为姿态 (roll, pitch, yaw)。
        Returns:
            int: 底层 `rm_movel` 函数的返回值，通常0表示成功。
        """
        if len(pose) != 6:
            raise ValueError("pose 参数必须包含6个元素: [x, y, z, roll, pitch, yaw]")
        
        # 使用 movel (笛卡尔空间线性移动) 命令机械臂移动
        # 参数: 目标位置姿态, 速度(%), 加速度(%), 运动半径(mm), 是否阻塞
        success = self.arm_controller.rm_movej_p(pose, 20, 0, 0, 1)
        return success
    
    def movej(self, pose):
        """
        以关节空间方式移动机械臂到指定的关节角度位置。

        Args:
            pose (list): 包含7个关节的目标绝对角度 (单位: 度)。

        Returns:
            int: 底层 `rm_movej` 函数的返回值，通常0表示成功。
        """
        if len(pose) != 7:
            raise ValueError("pose 参数必须包含7个元素，分别对应7个关节的角度")
        
        # 使用 movej (关节空间移动) 命令机械臂移动
        # 参数: 目标关节, 速度(%), 加速度(%), 运动半径(mm), 是否阻塞
        success = self.arm_controller.rm_movej(pose, 50, 0, 0, 1)
        return success

    def move(self, tech_state):
        """
        以增量方式移动机械臂。
        将输入的 `tech_state` (相对角度) 与当前关节角度相加，得到目标角度，
        然后命令机械臂移动到该目标位置。

        Args:
            tech_state (list): 包含7个关节的相对移动角度 (单位: 度)。

        Returns:
            int: 底层 `rm_movej` 函数的返回值，通常0表示成功。
        """
        self.prev_tech_state = self.get_state()
        print(f"当前状态: {self.prev_tech_state}")
        print(f"期望增量: {tech_state}")
        
        # 计算目标状态 = 当前状态 + 增量
        next_state = [self.prev_tech_state[i] + tech_state[i] for i in range(7)]
        
        print(f"目标状态: {next_state}")
        # 使用 movej (关节空间移动) 命令机械臂移动
        # 参数: 目标关节, 速度(%), 加速度(%), 运动半径(mm), 是否阻塞
        success = self.arm_controller.rm_movej(next_state, 50, 0, 0, 1)
        return success

    def move_check(self, state):
        """
        将机械臂移动到绝对关节角度位置。

        Args:
            state (list): 包含7个关节的目标绝对角度 (单位: 度)。

        Returns:
            int: 底层 `rm_movej` 函数的返回值。
        """
        success = self.arm_controller.rm_movej(state, 50, 0, 0, 1)
        return success

    def set_gripper(self, gripper_vari):
        """
        以增量方式设置夹爪的位置。

        Args:
            gripper_vari (float): 夹爪的相对移动量。正值表示张开，负值表示闭合。
                                  这个值的范围和单位需要根据实际模型调整。
                                  在此实现中，它被乘以1000来匹配底层接口。
        """
        self.pre_gripper = self.arm_controller.rm_get_gripper_state()
        print(f"当前夹爪状态: {self.pre_gripper}")
        
        # 计算新的目标位置
        new_pos = self.pre_gripper[1]["actpos"] + gripper_vari * 1000
        # 限制位置在 0 到 1000 之间
        new_pos = max(0, min(1000, new_pos))
        
        print(f"新的夹爪目标位置: {new_pos}")
        
        # 设置夹爪位置，参数: 位置(0-1000), 是否等待, 超时时间(ms)
        res = self.arm_controller.rm_set_gripper_position(int(new_pos), True, 10)
        print(f"设置夹爪结果: {res}")

    def __del__(self):
        """
        析构函数，在对象销毁时调用，用于释放资源。
        """
        try:
            if hasattr(self, 'arm_controller'):
                # 在这里可以添加断开连接或释放句柄的代码
                # self.arm_controller.rm_disconnect(self.handle)
                pass
        except Exception as e:
            print(f"销毁RM_controller时出错: {e}")

    def move_init(self, state):
        """
        将机械臂移动到初始绝对位置，通常用于启动时的复位。

        Args:
            state (list): 包含7个关节的目标绝对角度 (单位: 度)。

        Returns:
            int: 底层 `rm_movej` 函数的返回值。
        """
        # 使用较低的速度进行初始移动
        return self.arm_controller.rm_movej(state, 20, 0, 0, 1)

def initialize_robot(ip, mode=None):
    """
    封装了 `RoboticArm` 的初始化过程。

    Args:
        ip (str): 机械臂的IP地址。
        mode (rm_thread_mode_e, optional): 线程模式。

    Returns:
        tuple: 成功时返回 (RoboticArm实例, 句柄)，失败时返回None。
    """
    robot = RoboticArm(mode)
    # 创建机器人手臂实例并获取句柄
    handle = robot.rm_create_robot_arm(ip, 8080)
    if handle.id != -1: # 句柄ID不为-1表示成功
        return robot, handle
    return None

if __name__ == "__main__":
    # --- 测试代码 ---
    try:
        # 初始化控制器
        rm_controller = RM_controller("192.168.0.17", rm_thread_mode_e.RM_TRIPLE_MODE_E)
        
        # 测试获取状态
        current_state = rm_controller.get_state()
        print(f"成功获取机械臂状态: {current_state}")

        # 测试设置夹爪 (增量张开一点)
        # 先获取夹爪状态
        current_gripper = rm_controller.get_gripper()
        print(f"当前夹爪位置: {current_gripper}")
        print("\n测试设置夹爪...")
        rm_controller.set_gripper(-1.0) # 假设0.1表示张开10%
        time.sleep(0.5)
        current_gripper = rm_controller.get_gripper()
        print(f"设置后夹爪位置: {current_gripper}")

        # 测试增量移动
        print("\n测试增量移动...")
        # 仅移动第一个关节，增加5度
        move_increment = [5, 0, 0, 0, 0, 0, 0]
        rm_controller.move(move_increment)
        time.sleep(3)
        new_state = rm_controller.get_state()
        print(f"移动后机械臂状态: {new_state}")

    except ConnectionError as e:
        print(f"连接失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")