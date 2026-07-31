#!/bin/bash
# Copyright 2026 Tencent Inc.  All rights reserved.

# Use only one GPU for compilation.
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=${devices[0]}
fi

DIR=$1
OUTPUT_DIR=$2

# Hidden sizes + max block-residual banks per model.
#   Kimi-K3: H=7168, num_hidden_layers=93, attn_res_block_size=12
#            -> max_blocks (PP0) = ceil(47/12) = 4
#            -> max_blocks (PP1) = ceil(93/12) = 8
CONFIGS=(
    "7168 4"
    "7168 8"
)

for cfg in "${CONFIGS[@]}"; do
    read -r H MAX_BLOCKS <<< "$cfg"
    echo "Generating attn_res kernels for H=${H}, max_blocks=${MAX_BLOCKS}..."
    python "${DIR}/generate_attn_res.py" \
        --h "${H}" \
        --max-blocks "${MAX_BLOCKS}" \
        --output_dir "${OUTPUT_DIR}"
done

echo "Done. attn_res kernel artifacts written to ${OUTPUT_DIR}"
