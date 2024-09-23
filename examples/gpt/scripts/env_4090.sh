source /home/gehao/lhy/bashrc
conda activate hetu-py
source ../../../hetu.exp

export PATH="/usr/local/cuda-11.7/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH}"
export PATH="/root/anaconda3/envs/hetu-py/bin:${PATH}"

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=TIME
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_STRAGGLER=ANALYSIS

export HETU_MEMORY_PROFILE=WARN
export HETU_MAX_SPLIT_SIZE_MB=200
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20

export NCCL_DEBUG=VERSION

echo "env done"