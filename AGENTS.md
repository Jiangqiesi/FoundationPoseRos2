# AGENTS.md

## 项目概述

FoundationPoseROS2 — 基于 NVIDIA FoundationPose + SAM2 的 ROS2 6D 物体位姿估计与跟踪系统。采用前后端分离架构：感知服务端负责分割/跟踪，控制器负责机器人运动（MoveIt2 或直接 SDK），Python 客户端/CLI 封装所有 ROS2 Action/Service 调用。

## 构建与环境

```bash
# Conda 环境（Humble 用 Python 3.10，Foxy 用 3.8）
conda activate foundationpose_ros2
export PATH=/usr/local/cuda/bin:$PATH   # 设置 CUDA_HOME / nvcc

# 构建 C++ 扩展（FoundationPose CUDA 内核）
bash build_all_conda.sh

# 构建自定义 ROS2 消息（foundationpose_msgs）
cd foundationpose_msgs && colcon build
# 构建后需 source: source install/setup.bash
```

## 运行

```bash
# 1. 相机
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true \
  align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true

# 2. 感知服务端（默认 IDLE 状态，等待客户端触发分割）
python foundationpose_perception_server.py

# 3. 控制服务端（机械臂延迟连接，按需连接）
python foundationpose_realman_server.py --robot-ip 192.168.0.17 --objects 1 2

# 4. GUI 客户端（PySide6，交互式分割+分配+位姿可视化）
python foundationpose_gui_client.py --mesh-dir demo_data

# 5. 独立模式（无需服务端）
python foundationpose_ros_multi.py

# 6. 命令行工具
python foundationpose_cli.py resegment --force
python foundationpose_cli.py assign --meshes demo_data/ship/ship.obj --masks 0 --session-id 1
python foundationpose_cli.py pick --objects 1 2
python foundationpose_cli.py status
```

## 测试

```bash
# 运行全部测试
pytest

# 运行单个测试文件
pytest tests/test_cli_commands.py

# 运行单个测试用例
pytest tests/test_cli_commands.py::TestCLIHelp::test_main_help

# 详细输出
pytest -v

# pytest.ini 自动禁用 ROS/ament 插件：
#   -p no:launch_testing -p no:ament-copyright -p no:ament-flake8 等
```

测试文件位于 `tests/`。`conftest.py` 将项目根目录和编译后的 `foundationpose_msgs` 安装路径添加到 `sys.path`。依赖编译消息的测试使用 `@pytest.mark.skipif` 守卫。

### 测试约定
- 测试类按功能分组：`TestCLIHelp`、`TestFoundationPoseClientStructure`。
- 客户端 API 测试通过 `unittest.mock.patch` mock `rclpy`、`Node`、`ActionClient`、`MultiThreadedExecutor`。设置 `mock_rclpy.ok.return_value = False` 避免真实 rclpy 调用。
- CLI 测试以子进程方式运行脚本（`subprocess.run([sys.executable, CLI_SCRIPT, ...])`）— 仅冒烟测试（帮助文本、退出码）。
- 消息测试通过 `MSGS_AVAILABLE = os.path.exists(<install path>)` 和 `@pytest.mark.skipif(not MSGS_AVAILABLE, reason="foundationpose_msgs 未编译")` 守卫。

## 架构与关键文件

| 文件 | 职责 |
|---|---|
| `foundationpose_perception_server.py` | 感知服务端节点（SAM2 + FoundationPose），4态状态机（IDLE/SEGMENTING/AWAITING_ASSIGNMENT/TRACKING） |
| `foundationpose_moveit2_controller.py` | 基于 MoveIt2 的机器人控制器节点 |
| `foundationpose_realman_controller.py` | 直接 SDK 控制器节点 |
| `foundationpose_realman_server.py` | 组合控制器，含 Action/Service 服务端（MoveIt2 + SDK），机械臂延迟连接 |
| `foundationpose_client.py` | Python 客户端 API，封装所有 ROS2 Action/Service 调用，含后台图像订阅 |
| `foundationpose_cli.py` | 基于客户端 API 的命令行工具 |
| `foundationpose_gui_client.py` | PySide6 GUI 客户端，交互式分割+mask分配+位姿可视化 |
| `foundationpose_ros_multi.py` | 独立一体化节点（旧版入口） |
| `foundationpose_utils.py` | 配置加载器（`FoundationPoseConfig`）、TF 处理器 |
| `cam_2_base_transform.py` | 相机到基座坐标系变换工具 |
| `realman/RealMan.py` | RealMan 机械臂 SDK 封装（`RM_controller`） |
| `config/pick_place.yaml` | 中心配置：坐标系、话题、运动参数、物体抓取库 |
| `foundationpose_msgs/` | 自定义 ROS2 消息包 |

### 运行时拓扑
```
相机 → [感知服务端] → /Current_OBJ_position_X 话题
                     → /perception/* 服务
                     → /perception/masks_visualization 话题（分割后发布一次）
                     → /perception/masks_label 话题（分割后发布一次）
                     → /perception/pose_visualization 话题（跟踪时每帧发布）
       [控制器]     ← 订阅位姿话题
                     → /robot/pick_place action
                     → /controller/* 服务
       [客户端/CLI/GUI] ← 调用双端的 action + service + 订阅可视化话题
```

### ROS2 接口参考

Action `PickPlace`: Goal(`uint32[] object_ids`, `uint32 target_id`, `bool enable_grasp`, `float32 offset_z`, `float32 approach_distance`, `float32 lift_height`) → Result(`bool success`, `string message`, `uint32[] completed_objects`, `uint32[] failed_objects`) + Feedback(`string stage`, `uint32 current_object_id`, `uint32 objects_completed`, `uint32 objects_total`, `string detail`)

| 服务 | 请求 | 响应 |
|---|---|---|
| Resegment | `bool force` | `bool success`, `string message`, `uint32 session_id`, `uint32 num_masks` |
| AssignModels | `string[] mesh_paths`, `uint32[] mask_indices`, `uint32 session_id` | `bool success`, `string message` |
| GripperControl | `string action`, `float32 value` | `bool success`, `string message` |
| UpdateParams | `string param_name`, `string value` | `bool success`, `string message` |
| PerceptionStatus | `uint8 dummy` | `bool ready`, `bool segmentation_done`, `uint32 num_tracked_objects`, `uint32[] tracked_object_ids` |
| ControllerStatus | `uint8 dummy` | `bool robot_connected`, `bool moveit_ready`, `bool currently_executing`, `float32[] current_joints`, `string[] joint_names` |

### 默认服务/Action 名称（来自 `config/pick_place.yaml`）
- Action: `/robot/pick_place`
- 感知: `/perception/resegment`, `/perception/assign_models`, `/perception/get_status`
- 控制器: `/controller/gripper`, `/controller/update_params`, `/controller/get_status`
- 位姿话题: `/Current_OBJ_position_{id}`（历史命名，禁止修改）
- 可视化话题: `/perception/masks_visualization`（rgb8, transient_local）, `/perception/masks_label`（mono8, transient_local）, `/perception/pose_visualization`（rgb8）

## 代码风格与约定

### 语言与注释
- 双语代码库：注释和日志消息以**中文**（简体）为主。文档字符串中英混合。编辑时匹配所在文件的现有风格。
- 模块级文档字符串使用 `"""..."""`，附简短中文描述。

### Python 版本
- Python 3.10（ROS2 Humble）。禁止使用 3.11+ 特性。

### 导入
- 标准库 → 第三方库 → ROS2 → 本地模块。无严格强制，但遵循各文件中的现有模式。
- `sys.path.append('./FoundationPose')` 和 `sys.path.append('./FoundationPose/nvdiffrast')` 在模块顶部用于导入 FoundationPose 内部模块 — 禁止删除。
- `from estimater import *` 是有意为之（FoundationPose 上游代码）— 禁止重构。
- 部分文件存在重复的 `import os` — 无害，无需处理。
- 项目无 linter/formatter 配置（无 pyproject.toml、.flake8、.pylintrc、.editorconfig）。遵循现有文件风格。

### 类型标注
- 较新文件（`foundationpose_client.py`、`foundationpose_perception_server.py`、`foundationpose_realman_server.py`）使用 `typing` 标注：`Dict`、`List`、`Optional`、`Tuple`、`Callable`、`Any`。
- 旧文件无类型标注。新代码应添加标注；除非明确要求，不要为旧文件补充标注。
- `foundationpose_perception_server.py` 第 70 行有 pyright 抑制注释 — 这是有意为之，因 FoundationPose 的动态导入所需。

### 命名
- 类：`PascalCase`（`PoseEstimationNode`、`FoundationPoseClient`、`FileSelectorGUI`）
- 函数/方法：`snake_case`（`load_grasp_file`、`pose_callback`）
- ROS2 节点名：`snake_case` 字符串（`'foundationpose_moveit2_controller'`）
- ROS2 话题：`/Current_OBJ_position_X`（历史命名，保持不变）
- 配置键：YAML 中使用 `snake_case`

### 日志
- ROS2 节点：使用 `self.get_logger().info/warn/error()` — 节点代码中禁止使用 `print()`。
- CLI/交互脚本：`print()` 可用于面向用户的输出。
- 代码库中未使用 loguru。

### 错误处理
- 机械臂连接失败 → 返回错误字典（延迟连接模式，不再硬退出）
- 配置加载 → 警告并返回空字典（软失败）
- TF 查询 → 静默回退到硬编码变换矩阵
- ROS2 服务调用 → 超时后记录警告，返回错误字典
- MoveIt2 初始化失败 → 记录错误 + traceback，回退到直接 SDK 模式
- 禁止使用裸 `except:` — 至少使用 `except Exception as e:`

### 需保留的模式
- **猴子补丁**：`FoundationPose.__init__` 和 `register` 在模块级被补丁以添加 `is_register` 标志。服务端和独立模式文件中的实现完全一致。
- **上下文管理器**：`FoundationPoseClient` 支持 `with` 语句进行资源清理。
- **配置驱动**：服务名、话题前缀、运动参数均来自 `config/pick_place.yaml`，通过 `FoundationPoseConfig` 加载。
- **抓取库**：从 `demo_data/` 中的 YAML 文件加载，按物体 ID 索引。
- **动态消息路径**：`foundationpose_client.py` 和 `tests/conftest.py` 在运行时将 `foundationpose_msgs/install/.../site-packages` 插入 `sys.path`。
- **感知状态机**：`PerceptionState(IntEnum)` 4态（IDLE/SEGMENTING/AWAITING_ASSIGNMENT/TRACKING），默认 IDLE 只收相机数据不分割，客户端请求后才推进状态。
- **机械臂延迟连接**：`_ensure_robot_connected()` 在首次 pick_place/gripper 请求时才连接机械臂+初始化 MoveIt2，连接失败返回错误而非崩溃。
- **分割会话关联**：`session_id` 递增计数器，`Resegment` 响应返回 `session_id`，`AssignModels` 请求校验 `session_id` 一致性，防止旧分割结果被错误分配。

### 禁止事项
- 添加新 pip 依赖前必须先检查 `requirements.txt`。
- 禁止修改 `FoundationPose/` 下的文件 — 这是上游 vendored 代码。
- 禁止修改 ROS2 话题名（`/Current_OBJ_position_X`）— 下游消费者依赖这些名称。
- 禁止删除 `sys.path.append` — FoundationPose 导入所必需。
- 禁止使用 `numpy>=2` — 已锁定 `numpy<2` 以保证兼容性。
- 禁止使用 `# type: ignore`、`as any` 等方式抑制类型错误，除非匹配 `foundationpose_perception_server.py` 中现有的 pyright 注释模式。

### 文件组织
- 所有 Python 入口文件位于项目根目录（无 `src/` 目录）。非标准 ROS2 Python 包布局 — 直接运行脚本（`python script.py`），而非 `ros2 run`。
- 测试位于 `tests/`，`conftest.py` 负责路径设置。
- 配置位于 `config/`。
- 机械臂 SDK 封装位于 `realman/`。
- 自定义消息位于 `foundationpose_msgs/`（colcon 包）。
- MoveIt2 配置位于 `rm_moveit_config/`（colcon 工作空间，含 URDF/SRDF/launch）。
- 网格数据位于 `demo_data/`（已 gitignore）。
- 启动脚本位于根目录。
- 无 CI 流水线（.github/workflows 不存在）。
