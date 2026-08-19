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

# Kimi-K3: global num_heads=96, head_dim=128 (96/48/24/12).
# Support TP8.
# Kimi-Linear: global num_heads=32, head_dim=128 (32/16/8/4).
# Support TP1 and TP2.
# Keep both model families in one artifact set so runtime dispatch is explicit
# and no model can silently fall back to an incompatible local-head kernel.
python "${DIR}/generate_kda.py" \
    --num-heads 12 16 32  \
    --max-verify-tokens 128 \
    --ssm-dtype ${SSM_DTYPE} \
    --output_dir "${OUTPUT_DIR}"

echo "Done. KDA kernel artifacts written to ${OUTPUT_DIR}"
