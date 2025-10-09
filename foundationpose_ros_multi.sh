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
source /opt/ros/humble/setup.bash
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}~
python foundationpose_ros_multi.py