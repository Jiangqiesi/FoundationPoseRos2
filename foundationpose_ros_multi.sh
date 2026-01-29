#!/bin/bash
__conda_setup="$('/home/ym/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ym/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ym/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ym/miniconda3/bin:$PATH"
    fi
fi
conda activate foundationpose_ros
# 添加这一行,使用系统的 libstdc++
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}~
python foundationpose_ros_multi.py