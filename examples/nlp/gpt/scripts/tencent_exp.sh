#!/bin/bash

# bash scripts/hot_switch_multi.sh 7b two bf16 greedy host01

MODEL_SIZE=${1:-'7b'}
SWITCH=${2:-0}
HOSTFILE=${3:-'hostfile01'}
SEQ_LEN=${4:-1024}
GLOBAL_BATCH_SIZE=${5:-256}
MICRO_BATCH_SIZE=${6:-4}
DP=${7:-2}
TP=${7:-4}
PP=${7:-2}
HETERO=true

if [ "${MODEL_SIZE}" = "7b" ]; then
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "13b" ]; then
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_HEADS=40
elif [ "${MODEL_SIZE}" = "30b" ]; then
    # actually 30b = 12*num_layers*hidden_size^2
    NUM_LAYERS=60
    HIDDEN_SIZE=6528 #6672
    NUM_HEADS=48 # should be divided by tp32... so 48 will cause error!!!
elif [ "${MODEL_SIZE}" = "32b" ]; then
    NUM_LAYERS=60
    HIDDEN_SIZE=6656 #6672
    NUM_HEADS=64
else
    echo the model should be 7b/13b/30b for test.
    exit 0
fi

NNODES=$(cat ${HOSTFILE} | wc -l)
NUM_GPUS_PER_NODE=$( cat $HOSTFILE | head -n 1 | awk -F 'slots=' '{print $2}' )
WORLD_SIZE=$(( ${NNODES} * ${NUM_GPUS_PER_NODE} ))

# before
BEFORE_LAYERS_NUM_LIST="16,16,8,24"
BEFORE_MICRO_BATCH_NUM_LIST="[32,32]"
BEFORE_UNUSED_RANK="[0,1]"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:4,5:5,6:6,7:7,8:0,9:1,10:10,11:11,12:12,13:13,14:14,15:15}"
# BEFORE_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus 16 \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero \
    --hetero_layers $BEFORE_LAYERS_NUM_LIST \
    --rank_to_device_mapping $BEFORE_RANK_TO_DEVICE_MAPPING \
    --unused_rank $BEFORE_UNUSED_RANK \
    --file_name "before.json"

# after
AFTER_LAYERS_NUM_LIST="2,30,20,12"
AFTER_MICRO_BATCH_NUM_LIST="[20,44]"
AFTER_UNUSED_RANK="[0,2,3,6,7,9,11]"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"

AFTER_LAYERS_NUM_LIST="32,0,20,12"
AFTER_MICRO_BATCH_NUM_LIST="[20,44]"
AFTER_UNUSED_RANK="[0,1,6,7,8,9,10,11]"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus 16 \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero \
    --hetero_layers $AFTER_LAYERS_NUM_LIST \
    --rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
    --unused_rank $AFTER_UNUSED_RANK \
    --file_name "after.json"

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

export PATH=/jizhicfs/hymiezhao/miniconda3/envs/hetu-py/bin:$PATH
export HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../../" && pwd )"
export LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${HETU_HOME}/build/hetu/third_party/cutlass/install:${LD_LIBRARY_PATH}"
export PYTHONPATH="${HETU_HOME}/python_refactor:${HETU_HOME}/build/lib:${PYTHONPATH}"

echo HETU_HOME = $HETU_HOME

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=TIME
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_STRAGGLER=ANALYSIS
export HETU_MEMORY_PROFILE=MICRO_BATCH

export HETU_MAX_SPLIT_SIZE_MB=200
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20

export NCCL_DEBUG=WARN

file="straggler_exp/${MODEL_SIZE}_gpus${WORLD_SIZE}_result.txt"
echo will write result into ${file}...
dir=$(dirname "$file")
if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
fi
if [ ! -f "$file" ]; then
    touch "$file"
fi
echo -n > "$file"

if [ "${SWITCH}" = 1 ]; then
    mpirun --allow-run-as-root -np ${WORLD_SIZE} --hostfile ${HOSTFILE} \
        --bind-to none --map-by slot \
        --mca btl_tcp_if_include bond1 -x NCCL_SOCKET_IFNAME=bond1 \
        --mca oob_tcp_if_include bond1 \
        -x UCX_NET_DEVICES=bond1 -x NCCL_IB_DISABLE=0 -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_CUDA_SUPPORT=1  -x NCCL_DEBUG=VERSION \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NCCL_NVLS_ENABLE=0 -x NCCL_NET_GDR_READ=1 -x NCCL_SOCKET_NTHREADS=8 \
        -x NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 \
        -x NCCL_COLLNET_ENABLE=0  -x SHARP_COLL_ENABLE_SAT=0 -x NCCL_NET_GDR_LEVEL=2 -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_IB_TC=160 -x NCCL_PXN_DISABLE=0 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER -x HETU_MEMORY_PROFILE \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_switch.py \
        --num_strategy=2 \
        --ds_parallel_config ds_parallel_config/hetero/before.json,ds_parallel_config/hetero/after.json \
        --global_batch_size $GLOBAL_BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE \
        --json_file $JSON_FILE \
        --json_key $JSON_KEY \
        --vocab_file $VOCAB_FILE \
        --merge_file $MERGE_FILE \
        --vocab_size 30592 \
        --hidden_size $HIDDEN_SIZE \
        --num_hidden_layers $NUM_LAYERS \
        --num_attention_heads $NUM_HEADS \
        --seq_length $SEQ_LEN \
        --epochs 4 \
        --steps 40 \
        --lr 1e-4 \
        --adam_weight_decay 0.01 \
        --hidden_act relu \
        --dropout_prob 0.1 \
        --bf16 \
        --use_flash_attn \
        --use_two_node \
        --switch $SWITCH \
        --hetero_stage_gpus $TP \
        --hetero_pipeline \
        --hetero_data \
        --before_micro_batch_num_list $BEFORE_MICRO_BATCH_NUM_LIST \
        --before_rank_to_device_mapping $BEFORE_RANK_TO_DEVICE_MAPPING \
        --before_unused_rank $BEFORE_UNUSED_RANK \
        --after_micro_batch_num_list $AFTER_MICRO_BATCH_NUM_LIST \
        --after_rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
        --after_unused_rank $AFTER_UNUSED_RANK \
        --tencent
else
    mpirun --allow-run-as-root -np ${WORLD_SIZE} --hostfile ${HOSTFILE} \
        --bind-to none --map-by slot \
        --mca btl_tcp_if_include bond1 -x NCCL_SOCKET_IFNAME=bond1 \
        --mca oob_tcp_if_include bond1 \
        -x UCX_NET_DEVICES=bond1 -x NCCL_IB_DISABLE=0 -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_CUDA_SUPPORT=1  -x NCCL_DEBUG=VERSION \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NCCL_NVLS_ENABLE=0 -x NCCL_NET_GDR_READ=1 -x NCCL_SOCKET_NTHREADS=8 \
        -x NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 \
        -x NCCL_COLLNET_ENABLE=0  -x SHARP_COLL_ENABLE_SAT=0 -x NCCL_NET_GDR_LEVEL=2 -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_IB_TC=160 -x NCCL_PXN_DISABLE=0 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER -x HETU_MEMORY_PROFILE \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_pack_or_pad.py \
        --num_strategy=2 \
        --ds_parallel_config ds_parallel_config/hetero/before.json,ds_parallel_config/hetero/after.json \
        --global_batch_size $GLOBAL_BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE \
        --json_file $JSON_FILE \
        --json_key $JSON_KEY \
        --vocab_file $VOCAB_FILE \
        --merge_file $MERGE_FILE \
        --vocab_size 30592 \
        --hidden_size $HIDDEN_SIZE \
        --num_hidden_layers $NUM_LAYERS \
        --num_attention_heads $NUM_HEADS \
        --seq_length $SEQ_LEN \
        --epochs 4 \
        --steps 40 \
        --lr 1e-4 \
        --adam_weight_decay 0.01 \
        --hidden_act relu \
        --dropout_prob 0.1 \
        --bf16 \
        --use_flash_attn \
        --use_two_node \
        --switch $SWITCH \
        --hetero_stage_gpus $TP \
        --hetero_pipeline \
        --hetero_data \
        --micro_batch_num_list $AFTER_MICRO_BATCH_NUM_LIST \
        --rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
        --unused_rank $AFTER_UNUSED_RANK \
        --tencent
fi
           