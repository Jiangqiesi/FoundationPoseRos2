# FoundationPoseROS2 前后端分离架构 — 使用指南

## 概述

本项目将 FoundationPoseROS2 重构为**持久化服务端 + 轻量客户端**架构。服务端启动后常驻运行，客户端按需连接触发操作，无需每次测试都重启 FoundationPose 和 MoveIt2。

### 架构对比

| 之前 | 之后 |
|------|------|
| 每次测试重启 FoundationPose（加载模型数十秒） | 服务端常驻，模型只加载一次 |
| 每次测试重启 MoveIt2 + 机械臂连接 | 控制服务端常驻，保持连接 |
| stdin 手动输入命令 | CLI / Python API 远程触发 |
| Tkinter / OpenCV 阻塞式 GUI | 全自动分割+分配，可选 Service 覆盖 |

### 新增文件一览

| 文件 | 作用 |
|------|------|
| `foundationpose_msgs/` | ROS2 自定义消息包（1 个 Action + 6 个 Service） |
| `foundationpose_perception_server.py` | 感知服务端 — 无 GUI，自动分割+跟踪+Service 接口 |
| `foundationpose_realman_server.py` | 控制服务端 — PickPlace Action Server + Service 接口 |
| `foundationpose_client.py` | Python API 客户端模块 |
| `foundationpose_cli.py` | CLI 命令行客户端 |
| `start_servers.sh` | 一键启动脚本 |
| `config/pick_place.yaml` | 新增 `server` 配置节（服务名定义） |

---

## 快速开始

### 前置条件

- 已完成原项目的环境配置（conda 环境、CUDA、ROS2 Humble）
- 已编译 FoundationPose C++ 扩展（`bash build_all_conda.sh`）
- 已编译 MoveIt2 工作空间（`cd rm_moveit_config && colcon build`）

### 第一步：编译消息包

首次使用需要编译 `foundationpose_msgs`：

```bash
conda activate foundationpose_ros2
source /opt/ros/humble/setup.bash
cd foundationpose_msgs
colcon build --packages-select foundationpose_msgs
```

编译成功后会在 `foundationpose_msgs/install/` 下生成 Python 绑定。后续使用无需重复编译。

### 第二步：启动服务端（3 个终端）

**终端 1 — 启动相机**

```bash
conda activate foundationpose_ros2
bash setup_camera.sh
```

**终端 2 — 一键启动服务端**

```bash
conda activate foundationpose_ros2
bash start_servers.sh --robot-ip 192.168.0.17
```

该脚本会依次启动：
1. MoveIt2 demo.launch.py（等待 5 秒初始化）
2. 感知服务端（自动加载模型 → SAM2 分割 → 分配掩码 → 持续跟踪）
3. 控制服务端（连接机械臂 → 初始化 MoveIt2 → 等待 Action 请求）

可选参数：

```bash
# 指定要跟踪的物体 ID
bash start_servers.sh --robot-ip 192.168.0.17 --objects "1 2 3"
```

启动成功后会打印：

```
==========================================
✓ 所有服务端已启动
==========================================

MoveIt2 Demo      PID: 12345
感知服务器       PID: 12346
控制服务器       PID: 12347

客户端命令示例:
  python foundationpose_cli.py status
  python foundationpose_cli.py pick --objects 1 2 --target 5
  python foundationpose_cli.py gripper open
  python foundationpose_cli.py resegment --force

按 Ctrl+C 优雅退出所有进程...
```

**终端 3 — 客户端操作**

```bash
conda activate foundationpose_ros2
source /opt/ros/humble/setup.bash
source foundationpose_msgs/install/setup.bash

# 现在可以反复执行客户端命令，无需重启服务端
python foundationpose_cli.py status
python foundationpose_cli.py pick --objects 1 2
```

---

## CLI 命令行客户端

`foundationpose_cli.py` 提供 6 个子命令，覆盖所有操作场景。

### pick — 抓取放置

```bash
# 抓取物体 1 和 2
python foundationpose_cli.py pick --objects 1 2

# 抓取物体 1，放置到物体 5 的位置
python foundationpose_cli.py pick --objects 1 --target 5

# 自定义运动参数
python foundationpose_cli.py pick --objects 1 2 \
    --offset-z 0.15 \
    --approach-distance 0.12 \
    --lift-height 0.08
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--objects` | int 列表 | 必填 | 要抓取的物体 ID |
| `--target` | int | 0 | 放置目标 ID（0 表示不放置） |
| `--enable-grasp` | bool | True | 是否启用夹爪控制 |
| `--offset-z` | float | 0.133 | Z 轴放置偏移（米） |
| `--approach-distance` | float | 0.1 | 靠近物体时的距离（米） |
| `--lift-height` | float | 0.1 | 提升高度（米） |

执行过程中会实时打印进度反馈：

```
[approaching] 物体 1 (0/2 完成): 正在靠近物体
[descending] 物体 1 (0/2 完成): 正在下降
[closing_gripper] 物体 1 (0/2 完成): 关闭夹爪
[lifting] 物体 1 (0/2 完成): 提升物体
[approaching] 物体 2 (1/2 完成): 正在靠近物体
...

结果: 抓取放置完成
成功: [1, 2]
```

按 `Ctrl+C` 可随时取消当前任务（会安全停止并打开夹爪）。

### gripper — 夹爪控制

```bash
python foundationpose_cli.py gripper open    # 打开夹爪
python foundationpose_cli.py gripper close   # 关闭夹爪
```

### resegment — 重新分割

```bash
python foundationpose_cli.py resegment          # 普通重新分割
python foundationpose_cli.py resegment --force   # 强制重新分割
```

当场景中物体发生变化（新增/移除物体）时使用。服务端会重新运行 SAM2 分割并自动分配掩码。

### assign — 手动分配网格模型

```bash
python foundationpose_cli.py assign \
    --meshes demo_data/cup/cup.obj demo_data/box/box.stl \
    --masks 0 1
```

覆盖自动分配结果，手动指定网格文件与掩码索引的对应关系。`--meshes` 和 `--masks` 数量必须相同。

### status — 查询系统状态

```bash
python foundationpose_cli.py status
```

输出示例：

```json
{
  "perception": {
    "ready": true,
    "segmentation_done": true,
    "num_tracked_objects": 3,
    "tracked_object_ids": [1, 2, 3]
  },
  "controller": {
    "robot_connected": true,
    "moveit_ready": true,
    "currently_executing": false,
    "current_joints": [0.0, -0.5, 1.2, 0.0, 0.8, 0.0],
    "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
  }
}
```

### param — 更新运动参数

```bash
python foundationpose_cli.py param offset_z 0.15
python foundationpose_cli.py param approach_distance 0.12
python foundationpose_cli.py param lift_height 0.08
```

运行时动态修改控制参数，无需重启服务端。

---

## Python API 客户端

`foundationpose_client.py` 提供 `FoundationPoseClient` 类，适合集成到自定义脚本或 Jupyter Notebook 中。

### 基本用法

```python
from foundationpose_client import FoundationPoseClient

# 使用上下文管理器自动管理资源
with FoundationPoseClient() as client:
    # 等待服务端就绪
    client.wait_for_servers(timeout_sec=30.0)

    # 查询状态
    status = client.get_status()
    print(f"跟踪物体数: {status['perception']['num_tracked_objects']}")

    # 执行抓取
    result = client.pick_place(
        object_ids=[1, 2],
        target_id=5,
        offset_z=0.133
    )
    print(f"成功: {result['completed_objects']}")
    print(f"失败: {result['failed_objects']}")
```

### 带进度回调的抓取

```python
with FoundationPoseClient() as client:
    client.wait_for_servers()

    def on_feedback(feedback_msg):
        fb = feedback_msg.feedback
        print(f"[{fb.stage}] 物体 {fb.current_object_id} "
              f"({fb.objects_completed}/{fb.objects_total})")

    result = client.pick_place(
        object_ids=[1, 2, 3],
        feedback_callback=on_feedback
    )
```

### 完整 API 参考

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `wait_for_servers(timeout_sec=30.0)` | 等待所有服务端就绪 | `bool` |
| `get_status()` | 查询感知+控制器状态 | `dict` |
| `pick_place(object_ids, ...)` | 同步执行抓取放置 | `dict` |
| `pick_place_async(object_ids, ...)` | 异步执行抓取放置 | `Future` |
| `cancel_current_action()` | 取消当前 Action | `bool` |
| `open_gripper()` | 打开夹爪 | `dict` |
| `close_gripper()` | 关闭夹爪 | `dict` |
| `resegment(force=False)` | 重新分割物体 | `dict` |
| `assign_models(mesh_paths, mask_indices)` | 手动分配网格-掩码 | `dict` |
| `update_param(param_name, value)` | 更新运动参数 | `dict` |
| `shutdown()` | 关闭客户端 | `None` |

所有返回 `dict` 的方法都包含 `success` (bool) 和 `message` (str) 字段。

---

## 配置文件

`config/pick_place.yaml` 中新增了 `server` 配置节：

```yaml
server:
  perception:
    auto_assign: true          # 是否自动分配掩码给网格
    mesh_dir: demo_data        # 网格文件目录
    resegment_on_startup: true # 启动时是否自动分割
  controller:
    action_name: /robot/pick_place           # PickPlace Action 名称
    gripper_service: /controller/gripper     # 夹爪 Service 名称
    params_service: /controller/update_params # 参数更新 Service 名称
    status_service: /controller/get_status   # 控制器状态 Service 名称
  perception_services:
    resegment: /perception/resegment         # 重新分割 Service 名称
    assign_models: /perception/assign_models # 模型分配 Service 名称
    status: /perception/get_status           # 感知状态 Service 名称
```

客户端和服务端共享这些配置，修改服务名后两端会自动同步。

---

## ROS2 话题和接口

### 话题（与原项目一致）

| 话题 | 类型 | 说明 |
|------|------|------|
| `/Current_OBJ_position_<id>` | `PoseStamped` | 物体位姿（base_link 坐标系） |

### Action

| 名称 | 类型 | 说明 |
|------|------|------|
| `/robot/pick_place` | `PickPlace` | 抓取放置任务（支持进度反馈和取消） |

PickPlace Action 定义：

```
# Goal
uint32[] object_ids        # 要抓取的物体 ID 列表
uint32 target_id           # 放置目标 ID
bool enable_grasp          # 是否启用夹爪
float32 offset_z           # Z 轴偏移
float32 approach_distance  # 靠近距离
float32 lift_height        # 提升高度
---
# Result
bool success
string message
uint32[] completed_objects
uint32[] failed_objects
---
# Feedback
string stage               # 当前阶段
uint32 current_object_id
uint32 objects_completed
uint32 objects_total
string detail
```

Feedback 的 `stage` 字段取值：
- `opening_gripper` — 打开夹爪
- `approaching` — 靠近物体
- `descending` — 下降
- `closing_gripper` — 关闭夹爪
- `lifting` — 提升
- `moving_to_place` — 移动到放置位置
- `placing` — 放置

### Service

| 名称 | 类型 | 说明 |
|------|------|------|
| `/perception/resegment` | `Resegment` | 重新运行 SAM2 分割 |
| `/perception/assign_models` | `AssignModels` | 手动分配网格-掩码映射 |
| `/perception/get_status` | `PerceptionStatus` | 查询感知状态 |
| `/controller/gripper` | `GripperControl` | 夹爪开合控制 |
| `/controller/update_params` | `UpdateParams` | 运行时参数更新 |
| `/controller/get_status` | `ControllerStatus` | 查询控制器状态 |

---

## 典型工作流

### 场景一：重复测试抓取参数

```bash
# 服务端只启动一次
bash start_servers.sh --robot-ip 192.168.0.17

# 反复调整参数测试，无需重启
python foundationpose_cli.py pick --objects 1 --offset-z 0.13
python foundationpose_cli.py param offset_z 0.15
python foundationpose_cli.py pick --objects 1 --offset-z 0.15
python foundationpose_cli.py param approach_distance 0.08
python foundationpose_cli.py pick --objects 1
```

### 场景二：场景变化后重新分割

```bash
# 场景中物体发生变化
python foundationpose_cli.py resegment --force

# 查看新的跟踪状态
python foundationpose_cli.py status

# 继续抓取
python foundationpose_cli.py pick --objects 1 2 3
```

### 场景三：Python 脚本批量操作

```python
from foundationpose_client import FoundationPoseClient

with FoundationPoseClient() as client:
    client.wait_for_servers()

    # 依次抓取多个物体放到目标位置
    for obj_id in [1, 2, 3]:
        result = client.pick_place(
            object_ids=[obj_id],
            target_id=5
        )
        if not result['success']:
            print(f"物体 {obj_id} 抓取失败: {result['message']}")
            break

    # 完成后打开夹爪
    client.open_gripper()
```

---

## 故障排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| 客户端报 "服务端未在 30 秒内就绪" | 服务端未启动或未完成初始化 | 检查终端 2 的输出，等待 "所有服务端已启动" |
| 感知服务端启动后无跟踪 | 相机未启动或数据未就绪 | 确认终端 1 的相机节点正常运行 |
| 抓取失败 | 位姿过期或规划失败 | 用 `status` 检查跟踪状态，确认物体在视野内 |
| colcon build 失败 | ROS2 环境未 source | 先执行 `source /opt/ros/humble/setup.bash` |
| pytest 报 launch_testing 错误 | ROS2 插件冲突 | 项目已配置 `pytest.ini` 禁用冲突插件，使用 `pytest tests/` 即可 |

---

## 与原项目的兼容性

所有原有文件保持不变，原来的使用方式仍然有效：

```bash
# 原来的方式（仍然可用）
python foundationpose_ros_multi.py
python foundationpose_realman_place.py --robot-ip 192.168.0.17

# 新的方式（推荐）
bash start_servers.sh --robot-ip 192.168.0.17
python foundationpose_cli.py pick --objects 1 2
```
