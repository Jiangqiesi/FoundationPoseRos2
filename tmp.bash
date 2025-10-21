# 临时在当前 shell 中移除旧工作区路径
old="/home/ym/ws_moveit2/install"

for var in COLCON_PREFIX_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH PATH LD_LIBRARY_PATH PYTHONPATH; do
  # 取出变量值
  eval "val=\"\$$var\""
  if [ -n "$val" ]; then
    new=$(printf "%s\n" "$val" | tr ':' '\n' | grep -v -x "$old" | paste -sd':' -)
    # 导出回去（如果 new 为空就删除该变量）
    if [ -n "$new" ]; then
      export $var="$new"
    else
      unset $var
    fi
  fi
done

# 重新 source 系统 ROS（可选，但推荐）
source /opt/ros/humble/setup.bash

# 验证
echo "COLCON_PREFIX_PATH=[$COLCON_PREFIX_PATH]"
echo "AMENT_PREFIX_PATH=[$AMENT_PREFIX_PATH]"
