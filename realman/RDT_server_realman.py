#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3

RDT (Robotics Diffusion Transformer) 推理服务器脚本。

该脚本的功能:
1.  启动一个TCP/HTTP服务器，监听来自客户端的连接请求。
2.  加载预训练的RDT模型和语言指令嵌入。
3.  接收客户端发送的包含图像和机器人状态的JSON数据。
4.  解析HTTP请求，提取出图像(base64编码)和状态信息。
5.  将接收到的数据送入RDT模型进行推理，生成机械臂的下一步动作。
6.  将生成的动作打包成JSON格式，并通过HTTP响应发送回客户端。
7.  循环等待新的客户端请求。
"""
import os
import json
import argparse
import sys
import threading
import time
import yaml
from collections import deque
import numpy as np
import torch
from PIL import Image as PImage
import cv2
import socket
from scripts.agilex_model import create_model
import base64

# 定义模型期望的相机名称
CAMERA_NAMES = ['cam_wrist', 'cam_right_high', 'cam_left_high']

# 定义动作块大小，表示每N次请求重新进行一次完整的推理
ACTION_BLOCK = 4

# 全局变量，用于存储观测历史和语言指令嵌入
observation_window = None
lang_embeddings = None

# --- 模型初始化函数 ---

def make_policy(args):
    """根据配置创建并加载RDT策略模型。"""
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )
    return model

def set_seed(seed):
    """设置随机种子以保证可复现性。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_config(args):
    """生成模型所需的配置字典。"""
    return {
        'episode_len': args.max_publish_step,
        'state_dim': 16, # 状态维度：双臂各7关节+1夹爪
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }

# --- 数据处理与推理 ---

def update_observation_window(args, config, state, img_controller):
    """更新观测窗口，该窗口存储最近的观测数据作为模型输入。"""
    def base64_to_image(base64_string):
        """将base64编码的字符串解码为OpenCV图像。"""
        if base64_string is None: return None
        img_data = base64.b64decode(base64_string)
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    def jpeg_mapping(img):
        """模拟JPEG压缩，与训练过程保持一致。"""
        if img is None: return None
        img = cv2.imencode('.jpg', img)[1].tobytes()
        return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
        # 添加一个空的初始观测
        observation_window.append(
            {
                'qpos': None,
                'images': {name: None for name in config["camera_names"]},
            }
        )

    # 解码base64图像
    img_first = base64_to_image(img_controller['wrist'])
    img_right = base64_to_image(img_controller['right'])
    img_left = base64_to_image(img_controller['left'])

    # 应用JPEG映射
    img_first = jpeg_mapping(img_first)
    img_right = jpeg_mapping(img_right)
    img_left = jpeg_mapping(img_left)

    # 准备机器人状态数据
    qpos = torch.from_numpy(np.array(state)).float().cuda()
    
    # 添加新观测到窗口
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
    """执行一次完整的模型推理，生成一个动作序列。"""
    global observation_window, lang_embeddings

    time1 = time.time()
    # 从观测窗口提取图像和状态数据
    image_arrs = [
        observation_window[-2]['images'][name] for name in config['camera_names']
    ] + [
        observation_window[-1]['images'][name] for name in config['camera_names']
    ]
    images = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]
    proprio = observation_window[-1]['qpos'].unsqueeze(0)
    
    # 调用模型进行推理
    actions = policy.step(
        proprio=proprio,
        images=images,
        text_embeds=lang_embeddings
    ).squeeze(0).cpu().numpy()
    
    print(f"模型推理耗时: {time.time() - time1:.4f} 秒")
    return actions

# --- 网络与HTTP处理 ---

def parse_http_request(data):
    """一个简单的HTTP请求解析器，用于提取请求体(body)。"""
    try:
        header_end = data.find(b'\r\n\r\n')
        if header_end == -1: return None
        
        headers_part = data[:header_end].decode('utf-8')
        body = data[header_end+4:]
        
        content_length = 0
        for line in headers_part.split('\r\n'):
            if line.lower().startswith('content-length:'):
                content_length = int(line.split(':')[1].strip())
                break
        
        # 确保接收到了完整的请求体
        if len(body) < content_length: return None
        
        return body[:content_length].decode('utf-8')
    except Exception as e:
        print(f"HTTP请求解析失败: {e}")
        return None

def visualize_images(images_dict):
    """解码并显示接收到的base64图像，用于调试。"""
    def decode_and_show(name, b64_string):
        if b64_string:
            try:
                img_data = base64.b64decode(b64_string)
                img_np = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow(name, img)
            except Exception as e:
                print(f"解码或显示图像 {name} 失败: {e}")

    decode_and_show('Wrist Camera', images_dict.get('wrist'))
    decode_and_show('Right Camera', images_dict.get('right'))
    decode_and_show('Left Camera', images_dict.get('left'))
    cv2.waitKey(100)

if __name__ == '__main__':
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="RDT Inference Server")
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", help='模型配置文件的路径')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='预训练模型的名称或路径')
    parser.add_argument('--lang_embeddings_path', type=str, default="/home/ym/code/RoboticsDiffusionTransformer/outs/yanmai_task.pt", help='预编码语言指令嵌入的路径')
    parser.add_argument('--HOST', type=str, default="192.168.0.8", help='服务器监听的IP地址')
    parser.add_argument('--PORT', type=int, default=12345, help='服务器监听的端口')
    # 添加其他在原脚本中但此处未使用的参数，以保持兼容性
    parser.add_argument('--max_publish_step', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ctrl_freq', type=int, default=25)
    parser.add_argument('--chunk_size', type=int, default=64)
    args = parser.parse_args()

    # --- 初始化 ---
    if args.seed is not None:
        set_seed(args.seed)

    config = get_config(args)
    policy = make_policy(args) # 创建并加载模型
    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"正在运行指令: \"{lang_dict['instruction']}\" (来自: \"{lang_dict['name']}\")")
    lang_embeddings = lang_dict["embeddings"]
    
    # --- 服务器设置 ---
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((args.HOST, args.PORT))
    server_socket.listen(1)
    print(f"服务器正在 {args.HOST}:{args.PORT} 上监听...")

    actions = []
    index = 0
    batch = 3 # 每次从动作序列中取出3个动作进行加和

    # --- 服务器主循环 ---
    while True:
        try:
            print("等待客户端连接...")
            conn, addr = server_socket.accept()
            with conn:
                print(f"客户端 {addr} 已连接")
                try:
                    # 接收完整的HTTP请求数据
                    data = b''
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk: break
                        data += chunk
                        if b'\r\n\r\n' in data:
                            json_str = parse_http_request(data)
                            if json_str: break
                    
                    if not json_str:
                        print("未接收到完整数据，关闭连接")
                        continue
                    
                    data_dict = json.loads(json_str)
                    
                    # 提取数据
                    position = [float(x) for x in data_dict['position']]
                    img = data_dict['images']
                    visualize_images(img) # 可视化接收到的图像

                    print(f"接收到状态: {position}")

                    # 每 ACTION_BLOCK 次请求，重新进行一次完整的模型推理
                    if index % ACTION_BLOCK == 0:
                        state = np.array(position)
                        update_observation_window(args, config, state, img)
                        print("\n开始新的控制循环，正在生成动作序列...")
                        actions = inference_fn(args, config, policy)
                        print(f"成功生成动作序列，形状: {actions.shape}")

                    if actions is None or len(actions) == 0:
                        print("模型未生成有效动作。")
                        continue
                    
                    # 从动作序列中取出当前步的动作
                    start_idx = (index % ACTION_BLOCK) * batch
                    end_idx = start_idx + batch
                    action = np.sum(actions[start_idx:end_idx], axis=0)

                    # 准备发送回客户端的动作数据
                    action_dict = {f"left_joint{i+1}": float(action[i]) for i in range(7)}
                    action_dict["left_gripper"] = float(action[7])
                    action_dict.update({f"right_joint{i+1}": float(action[i+8]) for i in range(7)})
                    action_dict["right_gripper"] = float(action[15])
                    
                    # 构建并发送HTTP响应
                    response_body = json.dumps(action_dict)
                    response_headers = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: application/json\r\n"
                        f"Content-Length: {len(response_body)}\r\n"
                        "\r\n"
                    )
                    conn.sendall((response_headers + response_body).encode('utf-8'))
                    print(f"已发送动作数据: {action_dict}")
                    index += 1

                except json.JSONDecodeError:
                    print("数据格式错误,无法解析为JSON")
                except Exception as e:
                    print(f"处理请求时发生错误: {e}")
                finally:
                    conn.close()
                    print("连接已关闭。\n")
        except KeyboardInterrupt:
            print("服务器已停止")
            break
        except Exception as e:
            print(f"服务器发生严重错误: {e}")

    cv2.destroyAllWindows()