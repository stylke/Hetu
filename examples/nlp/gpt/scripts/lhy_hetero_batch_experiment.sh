#!/bin/bash

file="experiments/straggler/hetero_batch.txt"
dir=$(dirname "$file")
if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
fi
if [ ! -f "$file" ]; then
    touch "$file"
fi
echo -n > "$file"

NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-2048}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-1024}
GLOBAL_BATCH_SIZE=${5:-256}
MICRO_BATCH_SIZE=${6:-4}
DP=${7:-4}
TP=${8:-2}
PP=${9:-2}
HETERO_LAYER=${10:-2}
NUM_MICRO_BATCH=${11:-64} # should equal to GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

PATH="/home/pkuhetu/envs/miniconda3/envs/hetu-py/bin:${PATH}"
HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python_refactor:${HETU_HOME}/build/lib:${PYTHONPATH}"

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=WARN
export HETU_STRAGGLER=ANALYSIS

export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

# 生成hetero脚本
python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_gpus 16 \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero \
    --hetero_layers $HETERO_LAYER,$((NUM_LAYERS - HETERO_LAYER)),16,16,16,16,16,16

for i in $(seq 16 $NUM_MICRO_BATCH); do
    hetero_num_micro_batch=$((NUM_MICRO_BATCH - (DP - 1) * i))
    if [ $hetero_num_micro_batch -lt $PP ]; then
        continue
    fi
    # 运行
    echo "split num micro batch $hetero_num_micro_batch begin..."
    echo -e "\nsplit straggler num micro batch $hetero_num_micro_batch:" >> "$file"
    mpirun --allow-run-as-root -np 16 \
        -H job-26147b12-dd3f-4226-88a1-df64c6ec8ffa-master-0:8,job-26147b12-dd3f-4226-88a1-df64c6ec8ffa-worker-0:8 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_pack_or_pad.py \
            --num_strategy=2 \
            --ds_parallel_config ds_parallel_config/two_node/dp4_tp2_pp2.json,ds_parallel_config/hetero/dp4_tp2_pp2.json \
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
            --run_straggler_experiment \
            --hetero_stage_gpus 2 \
            --hetero_pipeline \
            --hetero_data \
            --normal_micro_batches $i \
            --straggler_file $file
done
