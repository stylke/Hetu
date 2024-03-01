# NCCL_DEBUG=info
NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-256}
GLOBAL_BATCH_SIZE=${5:-16}
NUM_MICRO_BATCHES=${6:-2}

PATH="/home/pkuhetu/envs/miniconda3/envs/hetu/bin:${PATH}"
HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python_refactor:${HETU_HOME}/build/lib:${PYTHONPATH}"

mpirun --allow-run-as-root -np 16 \
-H a100-0:8,a100-1:8 \
-x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
--output-filename logs/ds_parallel --merge-stderr-to-stdout \
python train_hetu_gpt_ds_parallel.py \
--ds_parallel_config ds_parallel_config/two_node/dp4_tp2_pp2.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--num_micro_batches $NUM_MICRO_BATCHES \
--dataset wikicorpus_en \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--seq_length $SEQ_LEN \
--epochs 20 \
--lr 1e-6 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--use_multi_node \
