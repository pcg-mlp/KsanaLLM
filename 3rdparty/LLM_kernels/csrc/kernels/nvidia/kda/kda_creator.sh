#!/bin/bash
# Copyright 2026 Tencent Inc.  All rights reserved.

# Use only one GPU for compilation.
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=${devices[0]}
fi

DIR=$1
OUTPUT_DIR=$2

# Recurrent-state dtypes to specialize the state kernels for. The runtime uses
# FP32; BF16 is built as well so that the alternative pool layout stays covered
# by the tests. Export KLLM_KDA_SSM_DTYPE="float32" to build a single dtype and
# roughly halve the state-kernel build.
SSM_DTYPE=${KLLM_KDA_SSM_DTYPE:-"float32 bfloat16"}

# Kimi-K3: global num_heads=96, head_dim=128.
# Generate local head counts for TP=1/2/4/8.
python "${DIR}/generate_kda.py" \
    --num-heads 96 48 24 12 \
    --max-verify-tokens 128 \
    --ssm-dtype ${SSM_DTYPE} \
    --output_dir "${OUTPUT_DIR}"

echo "Done. KDA kernel artifacts written to ${OUTPUT_DIR}"
