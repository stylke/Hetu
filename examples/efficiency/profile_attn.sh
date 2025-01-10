NUM_HEADS_VALUES=(4 8 16 32)
HEAD_DIM=128

source ./env.sh
for NUM_HEADS in "${NUM_HEADS_VALUES[@]}"; do
    for SEQ_LEN in $(seq 1024 1024 32768); do
        SAVE_FILE="./results/num_heads_${NUM_HEADS}.txt"
        SAVE_DIR=$(dirname "$SAVE_FILE")
        if [ ! -d "$SAVE_DIR" ]; then
            mkdir -p "$SAVE_DIR"
        fi
        if [ ! -e "$SAVE_FILE" ]; then
            > "$SAVE_FILE"
        fi
        echo "run exp: num_heads = ${NUM_HEADS}, seq_len = ${SEQ_LEN}"
        echo "seq len = ${SEQ_LEN}:" >> "$SAVE_FILE"
        python ./profile_attn.py \
            --save_file "${SAVE_FILE}" \
            --eager_device "cuda:7" \
            --seq_len ${SEQ_LEN} \
            --num_heads ${NUM_HEADS} \
            --head_dim ${HEAD_DIM}
    done
done
