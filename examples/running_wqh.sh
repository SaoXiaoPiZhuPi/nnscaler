#!/bin/bash


pushd /data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-new
export NNSCALER_HOME=$(pwd)
export PYTHONPATH=${NNSCALER_HOME}:$PYTHONPATH
popd

DTIME=`date +%m-%d`
MTIME=`date +%H-%M`
export PROFILE_OUTPUT=/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-new/examples/logs/${DTIME}
export date_str=$DTIME
export time_str=$MTIME
mkdir -p ${PROFILE_OUTPUT}

data=$(cat running_wqh.conf)
declare $data
for PLAN_NGPUS in 4;do
    export LOG_NAME=${PROFILE_OUTPUT}/${MTIME}.dp_size_$((${RUNTIME_NGPUS} / ${PLAN_NGPUS}))_${NAME}_${PLAN_NGPUS}.log
    if [ $# -le 1 ]; then
        echo "Node rank & Master Addr are not provided. Run on local single machine."
        if [ "$PIPLINE" == "True" ]; then
            torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} ${CODE} --run_mode compile --plan_ngpus ${PLAN_NGPUS} --runtime_ngpus ${RUNTIME_NGPUS} --name ${NAME}_${PLAN_NGPUS} --model_id ${MODEL_ID} --dataset_path ${DATASET_PATH} --explore_pipeline 2>&1 | tee ${LOG_NAME}
        else
            torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} ${CODE} --run_mode compile --plan_ngpus ${PLAN_NGPUS} --runtime_ngpus ${RUNTIME_NGPUS} --name ${NAME}_${PLAN_NGPUS} --model_id ${MODEL_ID} --dataset_path ${DATASET_PATH} 2>&1 | tee ${LOG_NAME}
        fi
    elif [ $# -eq 2 ]; then
        MASTER_ADDR=172.20.$1.2
        NODE_RANK=$2
        export LOG_NAME=${PROFILE_OUTPUT}/${MTIME}.dp_size_$((${RUNTIME_NGPUS} / ${PLAN_NGPUS}))_${NAME}_${NODE_RANK}.log
        if [ "$PIPLINE" == "True" ]; then
            torchrun --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} ${CODE} --plan_ngpus ${PLAN_NGPUS} --runtime_ngpus ${RUNTIME_NGPUS} --name ${NAME}_${PLAN_NGPUS} --model_id ${MODEL_ID} --dataset_path ${DATASET_PATH} --explore_pipeline 2>&1 | tee ${LOG_NAME}
        else
            torchrun --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} ${CODE} --plan_ngpus ${PLAN_NGPUS} --runtime_ngpus ${RUNTIME_NGPUS} --name ${NAME}_${PLAN_NGPUS} --model_id ${MODEL_ID} --dataset_path ${DATASET_PATH} 2>&1 | tee ${LOG_NAME}
        fi
        # export NODE_RANK=0;export MASTER_ADDR=172.20.1.2;torchrun --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK --nnodes=4 --nproc_per_node=8 train.py --run_mode compile --plan_ngpus 32 --runtime_ngpus 32 --model_id llama3.1_70b_model --dataset_path /data/haiqwa/zevin_nfs/dataset/bookcorpus_llama3_128K 2>&1 | tee compile.log
    else
        echo "Too much arguments"
    fi
done