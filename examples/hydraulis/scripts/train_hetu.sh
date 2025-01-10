NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
# HIDDEN_SIZE=${2:-256}
FFN_HIDDEN_SIZE=${3:-11008}
# FFN_HIDDEN_SIZE=${3:-2560}
NUM_HEADS=${4:-32}
GLOBAL_BATCH_SIZE=-1 # 目前改用gtn代替gbs
GLOBAL_TOKEN_NUM=${5:-10000}
MAX_SEQ_LEN=${6:-2048}
# SERVER_ADDR=${7:-"${IP_1}"} # master-0
SERVER_ADDR=${7:-"${IP_2}"} # worker-0
# SERVER_ADDR=${7:-"127.0.0.1"} 
SERVER_PORT=${8:-"23333"}
HOST_FILE_PATH=${9:-"${ENV_PATH}/host_single.yaml"}
ENV_FILE_PATH=${10:-"${ENV_PATH}/env_A100.sh"}
STRATEGY_POOL_PATH=${11:-"./strategy/strategy_pool_7b_A100.json"}

WARM_UP=0
COMPUTE_ONLY=0
TORCH_PROFILE=0
CASE=0
if [[ ${CASE} -eq 0 ]]; then
	# test
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], [(4, 2), (1, 8)], [(4, 2), (1, 4), (1, 4)]]"
	BATCHING_METHOD=4
	NUM_GPUS=8
	MULTI_TP_PP_LIST="[[(8, 1)], [(4, 1), (1, 4)]]"
	BATCHING_METHOD=4
elif [[ ${CASE} -eq 1 ]]; then
	# homo + greedy packing with static shape
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], ]"
	BATCHING_METHOD=2
elif [[ ${CASE} -eq 2 ]]; then	
    # homo + greedy packing with dynamic shape
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], ]"
	BATCHING_METHOD=3
elif [[ ${CASE} -eq 3 ]]; then	
    # homo + hydraulis packing
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], ]"
	BATCHING_METHOD=4
elif [[ ${CASE} -eq 4 ]]; then	
    # hetero + hydraulis packing
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (1, 8)], ]"
	BATCHING_METHOD=4
elif [[ ${CASE} -eq 5 ]]; then	
    # hetero + hydraulis packing
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (1, 4), (1, 4)], ]"
	BATCHING_METHOD=4
else
    echo unknown CASE
	exit 1
fi

echo num_gpus=${NUM_GPUS}, global_token_num = ${GLOBAL_TOKEN_NUM}, max_seq_len = ${MAX_SEQ_LEN}

if [[ ${NUM_LAYERS} -eq 32 && ${HIDDEN_SIZE} -eq 4096 && ${NUM_HEADS} -eq 32 ]]; then
	MODEL_SIZE=7b
	echo use llama 7b model...
elif [[ ${NUM_LAYERS} -eq 40 && ${HIDDEN_SIZE} -eq 5120 && ${NUM_HEADS} -eq 40 ]]; then
	MODEL_SIZE=13b
	echo use llama 13b model...
else
	MODEL_SIZE=-unknown-size
	echo use llama unknown-size model...
fi

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/case${CASE}/llama${MODEL_SIZE}_gpus${NUM_GPUS}_gtn${GLOBAL_TOKEN_NUM}_msl${MAX_SEQ_LEN}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
DATA_CACHE_PATH=${ROOT_FOLDER}/web
DATA_PATH=${ROOT_FOLDER}/web/web_content_document
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

if [ ! -d "ds_parallel_config" ]; then
  mkdir "ds_parallel_config"
fi

# compute-sanitizer can be added in front of python3 to check illegal mem access bug
CMD="python3 -u train_hetu.py \
--warm_up $WARM_UP \
--compute_only $COMPUTE_ONLY \
--torch_profile $TORCH_PROFILE \
--batching_method $BATCHING_METHOD \
--multi_tp_pp_list \"${MULTI_TP_PP_LIST}\" \
--global_batch_size $GLOBAL_BATCH_SIZE \
--global_token_num $GLOBAL_TOKEN_NUM \
--max_seq_len $MAX_SEQ_LEN \
--strategy_pool $STRATEGY_POOL_PATH \
--data_path $DATA_PATH \
--data_cache_path $DATA_CACHE_PATH \
--tokenizer_type "GPT2BPETokenizer" \
--split "98,1,1" \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 50304 \
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
# python3 ../../python/hetu/rpc/pssh_start.py \
python3 -m hetu.rpc.pssh_start \
--hosts ${HOST_FILE_PATH} \
--command "$CMD" \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS} \
--envs ${ENV_FILE_PATH} \
--log_path ${LOG_FOLDER}
