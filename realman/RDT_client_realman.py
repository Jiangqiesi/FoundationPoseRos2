# -*- coding: utf-8 -*-
"""
RDT 客户端脚本

该脚本负责:
1.  初始化并控制多台RealSense相机以捕获图像。
2.  初始化并控制两个RealMan机械臂。
3.  从相机和机械臂收集状态数据 (图像和关节角度)。
4.  将数据打包并通过HTTP请求发送到RDT推理服务器。
5.  接收服务器返回的动作指令。
6.  解析指令并控制机械臂执行动作。
7.  循环执行以上步骤, 实现闭环控制。
"""
import json
import os
import numpy as np
import base64
from PIL import Image
import requests
import io
import time
import cv2
import pyrealsense2 as rs
from Robotic_Arm.rm_robot_interface import *
from RealMan import RM_controller

# RealSense相机序列号配置
# 通过唯一的序列号来识别和区分不同的相机
CAMERA_SERIALS = {
    'head': '153122070447',        # 头顶相机
    'left_wrist': '427622270438',  # 左手腕相机
    'right_wrist': '427622270277', # 右手腕相机
}

# 初始化机械臂控制器
# 为左右两个机械臂分别创建控制器实例，并指定其IP地址
# 左臂使用三线程模式以提高性能
left_wrist_controller = RM_controller("192.168.0.18", rm_thread_mode_e.RM_TRIPLE_MODE_E)
right_wrist_controller = RM_controller("192.168.0.19")

def find_device_by_serial(serial_number):
    """
    根据序列号查找并返回一个RealSense设备。

    Args:
        serial_number (str): 要查找的设备的序列号。

    Returns:
        rs.device: 如果找到设备，则返回设备对象；否则返回None。
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial_number:
            return device
    return None

class Img_controller:
    """
    相机控制器类，用于管理多个RealSense相机。
    """
    def __init__(self):
        """
        初始化所有在 CAMERA_SERIALS 中定义的相机。
        """
        try:
            self.pipelines = {}
            self.configs = {}
            
            # 检查并打印所有可用的RealSense设备
            ctx = rs.context()
            devices = ctx.query_devices()
            print(f"找到 {len(devices)} 个 RealSense 设备")
            
            for i, device in enumerate(devices):
                try:
                    serial = device.get_info(rs.camera_info.serial_number)
                    name = device.get_info(rs.camera_info.name)
                    print(f"设备 {i}: {name} (SN: {serial})")
                except Exception as e:
                    print(f"获取设备 {i} 信息时出错: {e}")
            
            # 根据序列号初始化每个相机
            for camera_name, serial in CAMERA_SERIALS.items():
                try:
                    # 检查设备是否存在
                    device = find_device_by_serial(serial)
                    if device is None:
                        print(f"警告: 未找到序列号为 {serial} 的相机 {camera_name}")
                        continue
                    
                    # 创建pipeline和config
                    pipeline = rs.pipeline()
                    config = rs.config()
                    
                    # 启用指定序列号的设备
                    config.enable_device(serial)
                    
                    # 配置彩色图像流
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    
                    # 启动pipeline
                    pipeline.start(config)
                    
                    # 保存pipeline和config实例
                    self.pipelines[camera_name] = pipeline
                    self.configs[camera_name] = config
                    
                    print(f"成功初始化 {camera_name} 相机 (SN: {serial})")
                    
                except Exception as e:
                    print(f"启动 {camera_name} 相机失败: {str(e)}")
                    continue
            
            if len(self.pipelines) == 0:
                raise RuntimeError("没有相机被成功初始化")
                
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"初始化相机失败: {str(e)}")

    def get_img(self):
        """
        从所有已初始化的相机捕获图像。

        Returns:
            tuple: 包含头部、左手腕、右手腕相机的图像 (numpy.ndarray)。
                   如果某个相机图像获取失败，则对应值为None。
        """
        try:
            images = {}
            
            # 从每个相机的pipeline中获取帧
            for camera_name, pipeline in self.pipelines.items():
                try:
                    # 等待一帧数据，超时时间为1000毫秒
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    
                    # 获取彩色帧
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        print(f"从 {camera_name} 未获取到彩色帧")
                        continue
                    
                    # 将帧数据转换为numpy数组
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # RealSense默认输出BGR格式，转换为RGB以方便后续处理
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    
                    images[camera_name] = color_image
                    
                except Exception as e:
                    print(f"从 {camera_name} 获取帧时出错: {str(e)}")
                    images[camera_name] = None
                    continue
            
            # 按名称返回图像
            head_image = images.get('head', None)
            left_wrist_image = images.get('left_wrist', None)
            right_wrist_image = images.get('right_wrist', None)
            
            return head_image, left_wrist_image, right_wrist_image
            
        except Exception as e:
            raise RuntimeError(f"捕获图像时出错: {str(e)}")

    def cleanup(self):
        """
        停止所有相机pipeline，释放资源。
        """
        try:
            if hasattr(self, 'pipelines'):
                for camera_name, pipeline in self.pipelines.items():
                    try:
                        pipeline.stop()
                        print(f"已停止 {camera_name} 相机")
                    except Exception as e:
                        print(f"停止 {camera_name} 时出错: {str(e)}")
                print("所有相机 pipeline 已停止。")
        except Exception as e:
            print(f"清理资源时出错: {str(e)}")

    def __del__(self):
        """
        析构函数，在对象销毁时自动调用cleanup方法。
        """
        self.cleanup()

# 左臂初始位置关节角度 (单位: 度)
START_POSITION_ANGLE_LEFT_ARM = [85, -54, -6, -65, -29, -85, 80]
# 右臂初始位置关节角度 (单位: 度)
START_POSITION_ANGLE_RIGHT_ARM = [-74, 41, -4, 84, 46, 74, 103]

def initialize_realman():
    """
    初始化 RealMan 机械臂到预设的起始位置和夹爪状态。
    """
    global left_wrist_controller, right_wrist_controller

    # 设置左臂夹爪到最大位置 (张开)
    left_wrist_controller.set_gripper(1000)
    # 设置右臂夹爪到最大位置 (张开)
    right_wrist_controller.set_gripper(1000)

    # 注释掉移动到初始位置的代码，可以根据需要启用
    # left_signal = left_wrist_controller.move_init(START_POSITION_ANGLE_LEFT_ARM)
    # right_signal = right_wrist_controller.move_init(START_POSITION_ANGLE_RIGHT_ARM)
    
    # 等待机械臂动作完成
    time.sleep(2)
    print("机械臂夹爪初始化成功")
    return True

# 服务器配置
HOST = '192.168.0.8'  # RDT推理服务器的IP地址
PORT = 12345          # RDT推理服务器的端口
ADDR = (HOST, PORT)
# 任务指令，将发送给模型以指导其生成动作
instruction = "Use both hands to clamp the material tray and lift it up."

def start_client(img_controller: Img_controller, num: int, task: str = None):
    """
    客户端主循环函数。

    Args:
        img_controller (Img_controller): 相机控制器实例。
        num (int): 当前循环的序号，用于保存数据。
        task (str, optional): 任务指令。默认为None。
    """
    request_header = {'Content-Type': 'application/json'}
    request_id = ''
    
    try:
        # 1. 获取图像
        head_image, left_wrist_image, right_wrist_image = img_controller.get_img()
        
        # 可视化相机图像，用于调试
        if head_image is not None:
            cv2.imshow('Head Camera', cv2.cvtColor(head_image, cv2.COLOR_RGB2BGR))
        if left_wrist_image is not None:
            cv2.imshow('Left Wrist Camera', cv2.cvtColor(left_wrist_image, cv2.COLOR_RGB2BGR))
        if right_wrist_image is not None:
            cv2.imshow('Right Wrist Camera', cv2.cvtColor(right_wrist_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(100) # 短暂等待，让OpenCV窗口有时间刷新

        def encode_image(image):
            """将图像编码为base64字符串"""
            if image is None:
                return None
            # 将RGB图像转回BGR以进行OpenCV编码
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # 编码为.jpg格式
            _, buffer = cv2.imencode('.jpg', bgr_image)
            # 转换为base64字符串
            return base64.b64encode(buffer).decode('utf-8')

        # 2. 编码图像为base64
        wrist_base64 = encode_image(head_image)
        left_base64 = encode_image(left_wrist_image)
        right_base64 = encode_image(right_wrist_image)

        # 3. 获取机械臂状态
        try:
            # 获取原始状态数据
            position_left_raw = left_wrist_controller.get_state()
            position_right_raw = right_wrist_controller.get_state()
            gripper_left = left_wrist_controller.get_gripper()
            gripper_right = right_wrist_controller.get_gripper()
            
            # 打印原始数据用于调试
            print(f"左臂原始状态: {position_left_raw}")
            print(f"右臂原始状态: {position_right_raw}")
            
            def process_arm_data(raw_data):
                """处理从控制器获取的原始机械臂数据，提取关节角度列表。"""
                if isinstance(raw_data, dict):
                    # 尝试从字典中按常用键提取关节数据
                    for key in ['joint', 'angles', 'pose']:
                        if key in raw_data:
                            return list(raw_data[key])
                    # 如果找不到特定键，则使用字典的值
                    return list(raw_data.values())[:7]
                elif isinstance(raw_data, (list, tuple)):
                    return list(raw_data)
                else:
                    # 如果数据类型未知，返回默认值
                    print(f"警告: 未知的机械臂状态类型: {type(raw_data)}")
                    return [0.0] * 7

            position_left = process_arm_data(position_left_raw)
            position_right = process_arm_data(position_right_raw)
            
            # 确保关节数据长度为7，并转换为float类型
            position_left = [float(x) for x in position_left[:7]]
            position_right = [float(x) for x in position_right[:7]]
            gripper_left = float(gripper_left) if gripper_left is not None else 0.0
            gripper_right = float(gripper_right) if gripper_right is not None else 0.0
            
            # 4. 组装16维的状态向量
            # 格式: [左臂7关节, 左夹爪1, 右臂7关节, 右夹爪1]
            position = position_left + [gripper_left] + position_right + [gripper_right]
            
            print(f"最终状态向量: {position}")
            
            # 验证向量长度
            if len(position) != 16:
                print(f"警告: 状态向量长度为 {len(position)}，应为16。将进行填充或截断。")
                position = (position + [0.0] * 16)[:16]
            
        except Exception as e:
            print(f"获取机械臂状态时出错: {e}")
            # 出错时使用全零的默认状态
            position = [0.0] * 16

        # 5. 组装并发送数据到服务器
        send_msg = {
            'request_id': request_id,
            'position': position,
            'images': {
                'wrist': wrist_base64, # 头顶相机
                'right': right_base64, # 右手腕相机
                'left': left_base64    # 左手腕相机
            }
        }
        print('------ 正在发送信息 ------')
        print(f"发送的图像: "
              f"头顶-{'可用' if head_image is not None else '无'}, "
              f"左腕-{'可用' if left_wrist_image is not None else '无'}, "
              f"右腕-{'可用' if right_wrist_image is not None else '无'}")
        print(f"发送的状态: {position}")
        
        send_bytes = json.dumps(send_msg).encode('utf-8')
        url = f'http://{HOST}:{PORT}'
        
        try:
            # 发送POST请求
            response = requests.post(url=url, headers=request_header, data=send_bytes, timeout=30)
            
            # 6. 接收并处理服务器响应
            print(f"服务器响应状态: {response.status_code}")
            
            if response.status_code != 200:
                print(f"服务器返回错误状态: {response.status_code}")
                print(f"响应内容: {response.text}")
                return "0"
                
            action = response.text 
            print('------ 接收到动作 ------')
            print(action)

            try:
                action_dict = json.loads(action)
                
                # 提取左右臂的关节和夹爪动作
                left_joint = [float(action_dict[f"left_joint{i+1}"]) for i in range(7)]
                left_gripper = float(action_dict["left_gripper"])
                right_joint = [float(action_dict[f"right_joint{i+1}"]) for i in range(7)]
                right_gripper = float(action_dict["right_gripper"])

                # 7. 执行机械臂动作
                print(f"执行左臂动作: {left_joint}, 夹爪: {left_gripper}")
                print(f"执行右臂动作: {right_joint}, 夹爪: {right_gripper}")
                
                try:
                    left_wrist_controller.move(left_joint)
                    right_wrist_controller.move(right_joint)
                    left_wrist_controller.set_gripper(left_gripper)
                    right_wrist_controller.set_gripper(right_gripper)
                    print("机械臂动作执行成功")
                except Exception as e:
                    print(f"执行机械臂动作时出错: {e}")

                # 8. 保存接收到的动作数据
                try:
                    file_path = f"RDT_move_data/{num}.txt"
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(action)
                    print(f"数据已保存到 {file_path}")
                except Exception as e:
                    print(f"保存数据时出错: {e}")

                return action

            except (json.JSONDecodeError, KeyError) as e:
                print(f"解析或解析动作响应时出错: {e}")
                print(f"原始响应: {action}")
                return "0"
                
        except requests.exceptions.RequestException as e:
            print(f"HTTP请求失败: {e}")
            return "0"
            
    except Exception as e:
        print(f"客户端主循环出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return "0"

if __name__ == '__main__':
    # 初始化机械臂
    if not initialize_realman():
        exit(0)
    
    numnum = 0
    img_controller = None
    try:
        # 初始化相机控制器
        img_controller = Img_controller() 
        # 进入主循环
        while True:
            st = start_client(img_controller, num=numnum, task=instruction)
            if st == "0": # 表示出现错误或未收到有效动作
                time.sleep(1)
            elif st == "1": # 假设 "1" 是任务完成的信号
                print("自动化任务完成，请提取数据。")
                break
            else: # 成功执行一步，序号增加
                numnum += 1
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"主程序出错: {str(e)}")
    finally:
        # 确保资源被释放
        if img_controller:
            img_controller.cleanup()
        cv2.destroyAllWindows()
