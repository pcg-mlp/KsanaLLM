#!/bin/bash

# Use only one GPU for compilation.
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=${devices[0]}
fi

DIR=$1
OUTPUT_DIR=$2

# Model configs: name, H (hidden_size), I (inter_size)
MODELS=(
    "Flash 4096 2048"
    "Pro   7168 3072"
)

# Collect all N values to deduplicate.
declare -A N_SET
for model in "${MODELS[@]}"; do
    read -r name H I <<< "$model"
    for tp in 1 2 4 8; do
        for ep in 1 2 4 8; do
            moe_tp=$(( tp / ep ))
            if [ "$moe_tp" -lt 1 ]; then
                moe_tp=1
            fi
            N1=$H                                    # GEMM1: full hidden_size
            N2=$(( (I + moe_tp - 1) / moe_tp ))      # GEMM2: ceil(I / moe_tp)
            N_SET[$N1]=1
            N_SET[$N2]=1
        done
    done
done

# Generate cubin/json/ptx for each unique N.
for N in "${!N_SET[@]}"; do
    echo "Generating triton FP8 quant kernel for N=${N}..."
    python "${DIR}/generate_triton_fp8_quant.py" \
        --n "${N}" \
        --output_dir "${OUTPUT_DIR}"

    echo "Generating triton fused SiluAndMul+FP8 quant kernel for N=${N}..."
    python "${DIR}/generate_triton_silu_mul_quant.py" \
        --n "${N}" \
        --output_dir "${OUTPUT_DIR}"
done

echo "Done. Generated kernels for ${#N_SET[@]} N values."
