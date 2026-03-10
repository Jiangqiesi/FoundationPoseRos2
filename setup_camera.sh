#!/bin/bash
source /opt/ros/humble/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  enable_rgbd:=true enable_sync:=true align_depth.enable:=true \
  enable_color:=true enable_depth:=true pointcloud.enable:=true \
  rgb_camera.color_profile:=1280x720x6 \
  depth_module.depth_profile:=1280x720x6