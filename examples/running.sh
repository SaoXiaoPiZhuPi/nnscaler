#!/bin/bash

pushd /data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-0.4
export NNSCALER_HOME=$(pwd)
export PYTHONPATH=${NNSCALER_HOME}:$PYTHONPATH
popd

DTIME=`date +%m-%d`
MTIME=`date +%m-%d-%H-%M`
export PROFILE_OUTPUT=/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-0.4/examples/logs/${DTIME}
mkdir -p ${PROFILE_OUTPUT}

data=$(cat running.conf)
declare $data
export LOG_NAME=${PROFILE_OUTPUT}/${MTIME}.plan_${PLAN_NGPUS}_run${RUNTIME_NGPUS}.log
if [ $# -le 1 ]; then
    echo "Node rank & Master Addr are not provided. Run on local single machine."
    if [ $PLAN_NGPUS -le 8 ]; then
        torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} ${CODE} --plan_ngpus ${PLAN_NGPUS} --runtime_ngpus ${RUNTIME_NGPUS} --name ${NAME} --model_id ${MODEL_ID} --dataset_path ${DATASET_PATH} 2>&1 | tee run.log
    fi
elif [ $# -eq 2 ]; then
    MASTER_ADDR=172.20.$1.2
    NODE_RANK=$2
    torchrun --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} ${CODE} --plan_ngpus ${PLAN_NGPUS} --runtime_ngpus ${RUNTIME_NGPUS} --name ${NAME} --model_id ${MODEL_ID} --pas_priority ${PAS_PRIORITY} --use_zero ${USE_ZERO} --dataset_path ${DATASET_PATH} 2>&1 | tee ${LOG_NAME}
else
    echo "Too much arguments"
fi