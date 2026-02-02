# 扩展 NCCL 端口范围并启用端口复用
export NCCL_NSOCKS_PERTHREAD=32
export NCCL_SOCKET_NTHREADS=4
export NCCL_MIN_NCHANNELS=16

# export NCCL_DEBUG=INFO

# 配置参数
nnodes=2
nproc_per_node=4
MASTER_PORT=29500  # 固定端口
start=0            # GPU 起始编号（假设用 GPU 0–3）

# 设置可见设备
CUDA_VISIBLE_DEVICES=$(seq -s, $start $((start + nproc_per_node - 1)))

# 启动单个通信组的 profiling
torchrun --master_addr=192.168.10.155 --master_port=$MASTER_PORT \
         --node_rank=$1 --nnodes=$nnodes --nproc_per_node=$nproc_per_node \
         comm_profile.py \
         --comm_profile_dir=./comm

