#!/bin/bash

# only use one GPU
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=${devices[0]}
fi

DIR=$1
OUTPUT_DIR=$2

Candidate_BLOCK_SIZE_M=(16 32 64)
Candidate_Types=("FP16" "BF16")
Candidate_Quant=("True" "False")
Candidate_GROUP_SIZE=(32 128)
Candidate_TOP_K=(6 8)

for BM in "${Candidate_BLOCK_SIZE_M[@]}"; do
    for CT in "${Candidate_Types[@]}"; do
        for CQ in "${Candidate_Quant[@]}"; do
            for GS in "${Candidate_GROUP_SIZE[@]}"; do
                for TK in "${Candidate_TOP_K[@]}"; do
                    python $DIR/fused_moe_gptq_int4_fp8_kernel.py \
                        --BLOCK_SIZE_M $BM --BLOCK_SIZE_N 128 --BLOCK_SIZE_K 128 --GROUP_SIZE_M 1 \
                        --MUL_ROUTED_WEIGHT "False" --top_k $TK \
                        --compute_type $CT \
                        --group_size $GS \
                        --quant_a_per_tensor $CQ \
                        --m 222 --n 7168 --k 2048 --num_experts 256 \
                        --output_dir $OUTPUT_DIR --tune

                    python $DIR/fused_moe_gptq_int4_fp8_kernel.py \
                        --BLOCK_SIZE_M $BM --BLOCK_SIZE_N 128 --BLOCK_SIZE_K 128 --GROUP_SIZE_M 1 \
                        --MUL_ROUTED_WEIGHT "True" --top_k 1 \
                        --compute_type $CT \
                        --group_size $GS \
                        --quant_a_per_tensor $CQ \
                        --m 222 --n 7168 --k 2048 --num_experts 256 \
                        --output_dir $OUTPUT_DIR --tune
                done
            done
        done
    done
done