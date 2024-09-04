NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
# HIDDEN_SIZE=${2:-256}
FFN_HIDDEN_SIZE=${3:-11008}
# FFN_HIDDEN_SIZE=${3:-2752}
NUM_HEADS=${4:-32}
GLOBAL_BATCH_SIZE=${5:-64}
SERVER_ADDR=${6:-"172.24.19.127"} # master-0
# SERVER_ADDR=${6:-"172.24.71.238"} # worker-0
# SERVER_ADDR=${6:-"127.0.0.1"} # 216
SERVER_PORT=${7:-"23459"}
HOST_FILE_PATH=${8:-"./scripts/host.yaml"}
ENV_FILE_PATH=${9:-"./scripts/env_A100.sh"}

NUM_GPUS=16
MULTI_TP_PP_LIST="[[(1, 8), (4, 2)], ]"

echo num_gpus=${NUM_GPUS}, global_batch_size = ${GLOBAL_BATCH_SIZE}

if [[ ${NUM_LAYERS} -eq 32 && ${HIDDEN_SIZE} -eq 4096 && ${NUM_HEADS} -eq 32 ]]; then
	MODEL_SIZE=7b
	echo use llama 7b model...
elif [[ ${NUM_LAYERS} -eq 40 && ${HIDDEN_SIZE} -eq 5120 && ${NUM_HEADS} -eq 40 ]]; then
	MODEL_SIZE=13b
	echo use llama 13b model...
else
	MODEL_SIZE=unknown-size
	echo use llama unknown-size model...
fi

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/gpus${NUM_GPUS}_${MODEL_SIZE}_gbs${GLOBAL_BATCH_SIZE}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

CMD="python3 -u train_hetu.py \
--multi_tp_pp_list \"${MULTI_TP_PP_LIST}\" \
--global_batch_size $GLOBAL_BATCH_SIZE \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--epochs 4 \
--steps 40 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS}"

source ${ENV_FILE_PATH}
if [ ${NUM_GPUS} -gt 8 ]; then
python3 ../../python_refactor/hetu/rpc/pssh_start.py \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
else
python3 ../../python_refactor/hetu/rpc/pssh_start.py \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
fi