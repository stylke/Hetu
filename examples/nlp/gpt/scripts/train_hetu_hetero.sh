NUM_LAYERS=${1:-32}
# HIDDEN_SIZE=${2:-4096}
HIDDEN_SIZE=${2:-512}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-1024}
GLOBAL_BATCH_SIZE=${5:-256}
MICRO_BATCH_SIZE=${6:-4}
# FFN_HIDDEN_SIZE=${7:-11008}
FFN_HIDDEN_SIZE=${7:-2752}
SERVER_ADDR=${8:-"172.24.183.105"} # master-0
SERVER_PORT=${9:-"23456"}
HOST_FILE_PATH=${10:-"./scripts/host.yaml"}
ENV_FILE_PATH=${11:-"./scripts/env.sh"}

CASE=2
if [[ ${CASE} -eq 1 ]]; then
	# 单机同构
	# setting 1
	NUM_GPUS=8
	DP=2
	TP=2
	PP=2
	HETERO=false
elif [[ ${CASE} -eq 2 ]]; then
	# 单机异构
	# setting 2
	NUM_GPUS=8
	DP=2
	TP=2
	PP=2
	HETERO=true
	LAYERS_NUM_LIST="18,14,8,24"
	STAGES_NUM_LIST="[2,2]"
	MICRO_BATCH_NUM_LIST="[30,34]"
	UNUSED_RANK="[4]"
	RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7}"
elif [[ ${CASE} -eq 3 ]]; then
	# 多机同构
	# setting 3
	NUM_GPUS=16
	DP=4
	TP=2
	PP=2
	HETERO=false
elif [[ ${CASE} -eq 4 ]]; then	
	# 多机异构
	# setting 4
	NUM_GPUS=16
	DP=2
	TP=2
	PP=4
	HETERO=true
	LAYERS_NUM_LIST="1,10,11,10,8,8,8,8"
	STAGES_NUM_LIST="[4,4]"
	MICRO_BATCH_NUM_LIST="[28,36]"
	UNUSED_RANK="[1]"
	RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:14,7:15,8:8,9:9,10:10,11:11,12:12,13:13,14:6,15:7}"
else
    echo unknown CASE
	exit 1
fi

echo dp=${DP}, tp=${TP}, pp=${PP}, num_gpus=${NUM_GPUS} 

if [[ ${NUM_LAYERS} -eq 32 && ${HIDDEN_SIZE} -eq 4096 && ${NUM_HEADS} -eq 32 ]]; then
	MODEL_SIZE=7b
	echo use gpt 7b model...
elif [[ ${NUM_LAYERS} -eq 40 && ${HIDDEN_SIZE} -eq 5120 && ${NUM_HEADS} -eq 40 ]]; then
	MODEL_SIZE=13b
	echo use gpt 13b model...
else
	MODEL_SIZE=unknown-size
	echo use gpt unknown-size model...
fi

if [ ${SEQ_LEN} -lt 1024 ]; then
	SEQ=$SEQ_LEN
else
	SEQ=$(( ${SEQ_LEN} / 1024 ))k
fi
echo use seq_len = ${SEQ}

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/gpus${NUM_GPUS}_${MODEL_SIZE}_seq${SEQ}_gbs${GLOBAL_BATCH_SIZE}_mbs${MICRO_BATCH_SIZE}_dp${DP}_tp${TP}_pp${PP}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

python ./ds_parallel_config/generate_gpt_3d_config.py \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--tp $TP \
	--pp $PP \
	--zero

CMD="python3 -u train_hetu_hetero.py \
--num_strategy=1 \
--ds_parallel_config ds_parallel_config/homo/dp${DP}_tp${TP}_pp${PP}.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
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
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS}"

if [ "${HETERO}" = true ]; then

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--tp $TP \
	--pp $PP \
	--zero \
	--hetero_layers $LAYERS_NUM_LIST \
	--hetero_stages $STAGES_NUM_LIST \
	--rank_to_device_mapping $RANK_TO_DEVICE_MAPPING \
	--unused_rank $UNUSED_RANK

CMD="python3 -u train_hetu_hetero.py \
--num_strategy=1 \
--ds_parallel_config ds_parallel_config/hetero/dp${DP}_tp${TP}_pp${PP}.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
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
--hetero_pipeline \
--hetero_data \
--hetero_stage_gpus ${TP} \
--hetero_stages \"${STAGES_NUM_LIST}\" \
--micro_batch_num_list \"${MICRO_BATCH_NUM_LIST}\" \
--rank_to_device_mapping \"${RANK_TO_DEVICE_MAPPING}\" \
--unused_rank \"${UNUSED_RANK}\" \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS}"

fi

source ./scripts/env.sh
if [ ${NUM_GPUS} -gt 8 ]; then
python3 ../../../python_refactor/hetu/rpc/pssh_start.py \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
else
python3 ../../../python_refactor/hetu/rpc/pssh_start.py \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
fi