NUM_HEADS_VALUES=(4 8 16 32)
HEAD_DIM=128
SEQ_LEN=1024

source ./env.sh
for NUM_HEADS in "${NUM_HEADS_VALUES[@]}"; do
    for PACKING_NUM in $(seq 1 1 32); do
        SAVE_FILE="./packing_1024/num_heads_${NUM_HEADS}.txt"
        SAVE_DIR=$(dirname "$SAVE_FILE")
        if [ ! -d "$SAVE_DIR" ]; then
            mkdir -p "$SAVE_DIR"
        fi
        if [ ! -e "$SAVE_FILE" ]; then
            > "$SAVE_FILE"
        fi
        echo "run packing exp: num_heads = ${NUM_HEADS}, seq_len = ${SEQ_LEN}, packing_num = ${PACKING_NUM}"
        echo "packing num = ${PACKING_NUM}:" >> "$SAVE_FILE"
        python ./profile_attn.py \
            --save_file "${SAVE_FILE}" \
            --eager_device "cuda:6" \
            --seq_len ${SEQ_LEN} \
            --num_heads ${NUM_HEADS} \
            --head_dim ${HEAD_DIM} \
            --packing_num ${PACKING_NUM}
    done
done