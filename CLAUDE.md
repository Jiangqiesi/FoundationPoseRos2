# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
FoundationPoseROS2 is a ROS2 package for 6D object pose estimation and tracking using NVIDIA's FoundationPose and SAM2 (Segment Anything Model 2). It provides a real-time framework with an interactive GUI for selecting objects to track.

## Build & Setup
- **Environment**: Requires Conda environment `foundationpose_ros` (Python 3.10 for Humble, 3.8 for Foxy).
- **Dependencies**: CUDA 12.x, ROS2, PyTorch.
- **Install Script**: `bash build_all_conda.sh` (builds bundled C++ extensions).
  - *Note*: Ensure `CUDA_HOME` or path is set correctly for `nvcc` before running.

## Running the System
1.  **Camera Node**:
    ```bash
    ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true
    ```
2.  **Pose Estimation**:
    ```bash
    conda activate foundationpose_ros
    export PATH=/usr/local/cuda-12.X/bin:$PATH  # Adjust for your CUDA version
    python foundationpose_ros_multi.py
    ```
3.  **MoveIt2 Integration Test**:
    Use `test_rm_moveit.sh` to launch a demo with the MoveIt2 controller. This script builds the workspace if needed, launches MoveIt2 in the background, and runs the controller with example grasp configurations.
    ```bash
    ./test_rm_moveit.sh
    ```
    Key arguments demonstrated in the script:
    - `--objects`: IDs of objects to track.
    - `--grasp-files`: Mapping of object IDs to specific YAML grasp files (e.g., `1:demo_data/cup/cup_test.yml`).
    - `--place-pose`: 6D pose for placement `[x y z rx ry rz]`.
    - `--home-pose`: 6D pose to return to after task `[x y z rx ry rz]`.

## Architecture
- **Core Node**: `PoseEstimationNode` in `foundationpose_ros_multi.py`.
- **Logic Flow**:
  1.  Subscribes to RGB/Depth images and CameraInfo.
  2.  On first run, uses SAM2 + Tkinter GUI (`FileSelectorGUI`) to select objects from `demo_data/`.
  3.  Initializes `FoundationPose` instances for selected objects.
  4.  Tracks objects frame-by-frame and publishes poses as `PoseStamped` topics (`/Current_OBJ_position_X`).
- **Control Integration**:
  - `foundationpose_moveit2_controller.py`: Integrates with MoveIt2 for path planning and obstacle avoidance.
  - `foundationpose_realman_controller.py`: Direct SDK control for RealMan robots.
  - `realman/RealMan.py`: Wrapper around the RealMan robot SDK.
- **Coordinate Systems**:
  - Internal processing in camera frame.
  - Output transforms can be converted to base frame using `cam_2_base_transform.py`.

## Code Structure & Patterns
- **Entry Point**: `foundationpose_ros_multi.py` - Contains the main ROS2 node and application logic.
- **FoundationPose Integration**: The code monkey-patches `FoundationPose.__init__` and `register` methods at runtime to add state flags (`is_register`).
- **Meshes**: Object models (.obj/.stl) are loaded from `demo_data/` recursively.
- **Grasping**: `load_grasp_library` loads pre-computed grasps from YAML files.
- **UI Blocking**: The initialization phase blocks ROS callbacks while waiting for user input via OpenCV/Tkinter.
- **Robot Control**: The controllers support both MoveIt2 planning and direct SDK control fallback.

## Key Files
- `foundationpose_ros_multi.py`: Main pose estimation node.
- `foundationpose_moveit2_controller.py`: MoveIt2-based robot controller node.
- `foundationpose_realman_controller.py`: Simple SDK-based robot controller node.
- `realman/RealMan.py`: Python wrapper for RealMan robot interface.
- `build_all_conda.sh`: Compilation of C++ extensions.
- `cam_2_base_transform.py`: Transformation logic between camera and robot base.
