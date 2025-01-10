conda activate hetu-py # your env here
source ../../hetu.exp

export PATH="/root/anaconda3/envs/hetu-py/bin:${PATH}" # your PATH here
export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=TIME
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_STRAGGLER=ANALYSIS

export HETU_MEMORY_PROFILE=WARN
# export HETU_MAX_SPLIT_SIZE_MB=200
# export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20
export HETU_MAX_SPLIT_SIZE_MB=10240
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=0
# export HETU_PRE_ALLOCATE_SIZE_MB=10000


# Using multi-stream cuda event to watch time elaspe is inaccurate!
# export HETU_PARALLEL_ATTN=ANALYSIS
export HETU_PARALLEL_ATTN_SPLIT_PATTERN=NORMAL

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=VERSION
export NCCL_SOCKET_IFNAME=en,eth,em,bond0
export GLOO_SOCKET_IFNAME=en,eth,em,bond0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_NET_GDR_READ=1
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0

echo "env done"
