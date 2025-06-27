#!/bin/bash

# 检查参数数量是否正确
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <machine_numbers> <script.sh>"
    echo "Example: $0 101 102 103 script.sh"
    exit 1
fi

# 获取脚本名
SCRIPT_NAME=${!#}

# 获取第一台机器的编号
FIRST_MACHINE_NUM=$1

# 初始化当前机器序号
CURRENT_INDEX=0

# 遍历所有机器编号
for ((i=1; i<$#; i++)); do
    MACHINE_NUM=${!i}
    IP="172.20.${MACHINE_NUM}.2"
    
    echo "Connecting to machine $MACHINE_NUM with IP $IP..."
    
    # 使用 nohup 在主控机上挂起 SSH 命令，并传递参数
    nohup ssh "$IP" "bash -s" < "$SCRIPT_NAME" "$FIRST_MACHINE_NUM" "$CURRENT_INDEX" > /dev/null 2>&1 &
    
    if [ $? -eq 0 ]; then
        echo "Script started successfully on machine $MACHINE_NUM (SSH PID: $!)."
        echo "running $SCRIPT_NAME $FIRST_MACHINE_NUM $CURRENT_INDEX"
    else
        echo "Failed to start script on machine $MACHINE_NUM."
    fi
    
    # 递增当前机器序号
    CURRENT_INDEX=$((CURRENT_INDEX + 1))
done

# 等待所有后台任务完成
wait
echo "All SSH sessions have completed."