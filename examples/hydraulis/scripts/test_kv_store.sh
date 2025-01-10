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

NUM_GPUS=8
LOG_FOLDER=logs/test_kv_store
mkdir -p ${LOG_FOLDER}

# compute-sanitizer can be added in front of python3 to check illegal mem access bug
CMD="python3 -u test_kv_store.py \
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
