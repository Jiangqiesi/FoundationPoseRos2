import os
import json
import time
import numpy as np
import cv2
import torch
import base64
import yaml
from collections import deque
from PIL import Image as PImage

# 导入项目特定的模块
import pyrealsense2 as rs
from Robotic_Arm.rm_robot_interface import *
from RealMan import RM_controller
from scripts.agilex_model import create_model

# 相机配置: 通过序列号指定要使用的RealSense相机
CAMERA_SERIALS = {
    'head': '153122070447',
    'left_wrist': '427622270438',
    'right_wrist': '427622270277',
}

# 机械臂控制器初始化: 连接到指定IP地址的左右机械臂
left_wrist_controller = RM_controller("192.168.0.18", rm_thread_mode_e.RM_TRIPLE_MODE_E)
right_wrist_controller = RM_controller("192.168.0.19")

def find_device_by_serial(serial_number):
    """根据序列号查找并返回RealSense设备对象。"""
    ctx = rs.context()
    devices = ctx.query_devices()
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial_number:
            return device
    return None

class Img_controller:
    """相机控制类，用于初始化和捕获多个RealSense相机的图像。"""
    def __init__(self):
        """初始化所有在CAMERA_SERIALS中定义的相机。"""
        self.pipelines = {}
        self.configs = {}
        ctx = rs.context()
        devices = ctx.query_devices()
        print(f"找到 {len(devices)} 个 RealSense 设备")
        for camera_name, serial in CAMERA_SERIALS.items():
            device = find_device_by_serial(serial)
            if device is None:
                print(f"警告: 未找到序列号为 {serial} 的相机 {camera_name}")
                continue
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            self.pipelines[camera_name] = pipeline
            self.configs[camera_name] = config
            print(f"成功初始化 {camera_name} 相机 (SN: {serial})")
        if len(self.pipelines) == 0:
            raise RuntimeError("没有相机被成功初始化")

    def get_img(self):
        """从所有相机捕获图像。"""
        images = {}
        for camera_name, pipeline in self.pipelines.items():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print(f"从 {camera_name} 未获取到彩色帧")
                    images[camera_name] = None
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                images[camera_name] = color_image
            except Exception as e:
                print(f"从 {camera_name} 获取帧时出错: {e}")
                images[camera_name] = None
        
        head_image = images.get('head', None)
        left_wrist_image = images.get('left_wrist', None)
        right_wrist_image = images.get('right_wrist', None)
        return head_image, left_wrist_image, right_wrist_image

    def cleanup(self):
        """停止所有相机pipeline，释放资源。"""
        for camera_name, pipeline in self.pipelines.items():
            try:
                pipeline.stop()
                print(f"已停止 {camera_name} 相机")
            except Exception as e:
                print(f"停止 {camera_name} 时出错: {str(e)}")
        print("所有相机 pipeline 已停止。")

    def __del__(self):
        """析构函数，确保在对象销毁时清理资源。"""
        self.cleanup()

# --- 模型相关函数 ---

def make_policy(args):
    """根据配置创建并返回RDT策略模型。"""
    with open(args['config_path'], "r") as fp:
        config = yaml.safe_load(fp)
    args['config'] = config
    # 指定预训练的视觉编码器
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args['config'],
        dtype=torch.bfloat16,  # 使用bfloat16以提高性能
        pretrained=args['pretrained_model_name_or_path'],
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args['ctrl_freq'],
        device='cuda:1'  # 将模型加载到指定的GPU设备
    )
    return model

def set_seed(seed):
    """设置随机种子以确保实验的可复现性。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_config(args):
    """生成并返回模型所需的配置字典。"""
    CAMERA_NAMES = ['cam_wrist', 'cam_right_high', 'cam_left_high']
    config = {
        'episode_len': args['max_publish_step'],
        'state_dim': 16,  # 状态维度 (双臂各7关节+1夹爪)
        'chunk_size': args['chunk_size'],
        'camera_names': CAMERA_NAMES,
    }
    return config

# 全局变量，用于存储观测历史和语言指令嵌入
observation_window = None
lang_embeddings = None

def update_observation_window(args, config, state, img_controller):
    """
    更新观测窗口，该窗口存储了最近的几次观测（图像和机器人状态），
    作为模型的输入。
    """
    def jpeg_mapping(img):
        """模拟训练时使用的JPEG压缩，以保持数据分布一致。"""
        if img is None: return None
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    global observation_window
    if observation_window is None:
        # 初始化观测窗口，使用deque可以自动管理窗口大小
        observation_window = deque(maxlen=2)
        # 添加一个空的初始观测
        observation_window.append(
            {
                'qpos': None,
                'images': {name: None for name in config["camera_names"]},
            }
        )

    # 获取并处理图像
    img_first = jpeg_mapping(img_controller['wrist'])
    img_right = jpeg_mapping(img_controller['right'])
    img_left = jpeg_mapping(img_controller['left'])

    # 将机器人状态转换为torch张量并移到GPU
    qpos = torch.from_numpy(np.array(state)).float().cuda()
    
    # 将新的观测添加到窗口中
    observation_window.append(
        {
            'qpos': qpos,
            'images': {
                config["camera_names"][0]: img_first,
                config["camera_names"][1]: img_right,
                config["camera_names"][2]: img_left,
            },
        }
    )

def inference_fn(args, config, policy):
    """
    执行一次模型推理，生成机械臂的动作。
    """
    global observation_window
    global lang_embeddings
    
    # 从观测窗口中提取最近两次的图像
    image_arrs = [
        observation_window[-2]['images'][name] for name in config['camera_names']
    ] + [
        observation_window[-1]['images'][name] for name in config['camera_names']
    ]
    
    # 将numpy数组转换为PIL图像
    images = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]
    
    # 获取最新的机器人状态
    proprio = observation_window[-1]['qpos'].unsqueeze(0) # 增加batch维度
    
    # 调用策略模型进行推理
    actions = policy.step(
        proprio=proprio,
        images=images,
        text_embeds=lang_embeddings
    ).squeeze(0).cpu().numpy() # 移除batch维度并转为numpy数组
    
    return actions

def visualize_images(images_dict):
    """使用OpenCV显示来自不同相机的图像以进行调试。"""
    for name, img in images_dict.items():
        if img is not None:
            cv2.imshow(f'{name} Camera', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(100)

def get_robot_state():
    """获取并整合左右机械臂的状态，返回一个16维的向量。"""
    try:
        position_left_raw = left_wrist_controller.get_state()
        position_right_raw = right_wrist_controller.get_state()
        gripper_left = left_wrist_controller.get_gripper()
        gripper_right = right_wrist_controller.get_gripper()

        def extract_position(raw):
            """从原始数据中提取关节位置列表。"""
            if isinstance(raw, dict):
                for k in ['joint', 'angles', 'pose']:
                    if k in raw: return raw[k]
                return list(raw.values())[:7]
            return list(raw) if isinstance(raw, (list, tuple)) else [0.0] * 7

        position_left = extract_position(position_left_raw)
        position_right = extract_position(position_right_raw)
        
        # 确保数据长度和类型正确
        position_left = [float(x) for x in (position_left[:7] + [0.0] * 7)[:7]]
        position_right = [float(x) for x in (position_right[:7] + [0.0] * 7)[:7]]
        gripper_left = float(gripper_left) if gripper_left is not None else 0.0
        gripper_right = float(gripper_right) if gripper_right is not None else 0.0

        # 组合成16维状态向量
        position = position_left + [gripper_left] + position_right + [gripper_right]
        if len(position) != 16:
            position = (position + [0.0] * 16)[:16]
            
    except Exception as e:
        print(f"获取机械臂状态时出错: {e}")
        position = [0.0] * 16 # 出错时返回零向量
    return position
 

if __name__ == '__main__':
    # --- 参数配置 ---
    args = {
        'max_publish_step': 10000,
        'seed': None,
        'config_path': "configs/base.yaml",
        'pretrained_model_name_or_path': "/home/ym/code/RoboticsDiffusionTransformer/checkpoints/pick_place_single_v1/checkpoint-15000",
        'ctrl_freq': 50,
        'chunk_size': 64,
    }
    if args['seed'] is not None:
        set_seed(args['seed'])

    # --- 初始化 ---
    config = get_config(args)
    policy = make_policy(args) # 创建模型
    # 加载预先编码好的语言指令嵌入
    lang_dict = torch.load("/home/ym/code/RoboticsDiffusionTransformer/outs/dualarm-rdt-pick-up-v1.pt")
    print(f"正在运行指令: "{lang_dict['instruction']}"")
    lang_embeddings = lang_dict["embeddings"]

    img_controller = Img_controller() # 初始化相机
    
    # --- 主循环变量 ---
    numnum = 0
    ACTION_BLOCK = 1 # 每次推理生成一个动作块
    batch = 1
    index = 0
    actions = []
    instruction = "Put the charger head, the toy and the blue bottle into the white box."

    # 可以取消注释以将机械臂移动到初始姿态
    # left_wrist_controller.move([49.95, -78.13, -8.65, -77.15, 97.61, 57.79, -102.29])
    # right_wrist_controller.move([-19.72,84.84,-4.63,83.35,-52.60,-51.63,-113.88])

    try:
        # --- 控制主循环 ---
        while True:
            # 1. 采集数据 (图像和机器人状态)
            head_image, left_wrist_image, right_wrist_image = img_controller.get_img()
            images = {
                'wrist': head_image,
                'right': right_wrist_image,
                'left': left_wrist_image
            }
            visualize_images(images) # 可视化图像
            position = get_robot_state()

            # 2. 模型推理
            if index % ACTION_BLOCK == 0:
                print(f"机器人状态 = {position}")
                state = np.array(position)
                update_observation_window(args, config, state, images)
                print("
开始控制循环...")
                actions = inference_fn(args, config, policy)
                print(f"生成的动作形状 = {actions.shape}")

            if actions is None or len(actions) == 0:
                print("RDT出错,未生成有效动作,请检查模型或输入数据。")
                continue

            # 提取当前时间步的动作
            action = np.sum(actions[0:batch], axis=0)

            # 3. 执行机械臂动作
            left_joint = [float(action[i]) for i in range(7)]
            left_gripper = float(action[7])
            right_joint = [float(action[i]) for i in range(8, 15)]
            right_gripper = float(action[15])
            print(f"左臂动作 = {left_joint}, 夹爪 = {left_gripper}")
            print(f"右臂动作 = {right_joint}, 夹爪 = {right_gripper}")

            try:
                left_wrist_controller.move(left_joint)
                right_wrist_controller.move(right_joint)
                left_wrist_controller.set_gripper(left_gripper)
                right_wrist_controller.set_gripper(right_gripper)
                print("机械臂动作执行成功")
            except Exception as e:
                print(f"执行机械臂动作时出错: {e}")

            # 4. 保存数据 (用于调试或分析)
            try:
                file_path = f"RDT_move_data/{numnum}.txt"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(json.dumps({
                        "left_joint": left_joint, "left_gripper": left_gripper,
                        "right_joint": right_joint, "right_gripper": right_gripper
                    }))
                print(f"数据已保存到 {file_path}")
            except Exception as e:
                print(f"保存数据时出错: {e}")

            numnum += 1
            index += 1
            time.sleep(1) # 控制循环频率

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"主程序出错: {str(e)}")
    finally:
        # 清理资源
        img_controller.cleanup()
        cv2.destroyAllWindows()