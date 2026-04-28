#!/bin/bash

# only use one GPU
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=${devices[0]}
fi

DIR=$1
OUTPUT_DIR=$2

PARAM_LIST=(
#   0 group_n
#   1 group_k
#   2 BLOCK_SIZE_M
#   3 BLOCK_SIZE_N
#   4 BLOCK_SIZE_K
#   5 GROUP_SIZE_M
#   6 num_warps
#   7 num_stages
#   8 top_k
#   9 compute_type
#   10 use_fp8_w8a8
#   11 use_int8_w8a16
#   12 m
#   13 n
#   14 k
#   15 num_experts
#   16 tune
#  Only DeepSeek-R1/V3 (topk=8) and DeepSeek-V2 (topk=6), BF16 only
#  0  1  2   3   4   5 6  7  8  9     10     11     12  13    14   15  16
  "0, 0, 64, 64, 32, 8,  ,  , 8, BF16, False, False,    ,     ,    ,    , --tune,"
  "0, 0, 64, 64, 32, 8,  ,  , 6, BF16, False, False,    ,     ,    ,    , --tune,"
)

for param in "${PARAM_LIST[@]}"
do
  IFS=',' read -ra param_arr <<< "${param// /}"
  cmd="python $DIR/fused_batched_moe.py --output_dir $OUTPUT_DIR"
  [[ -n ${param_arr[0]} ]]  && cmd+=" --group_n ${param_arr[0]}"
  [[ -n ${param_arr[1]} ]]  && cmd+=" --group_k ${param_arr[1]}"
  [[ -n ${param_arr[2]} ]]  && cmd+=" --BLOCK_SIZE_M ${param_arr[2]}"
  [[ -n ${param_arr[3]} ]]  && cmd+=" --BLOCK_SIZE_N ${param_arr[3]}"
  [[ -n ${param_arr[4]} ]]  && cmd+=" --BLOCK_SIZE_K ${param_arr[4]}"
  [[ -n ${param_arr[5]} ]]  && cmd+=" --GROUP_SIZE_M ${param_arr[5]}"
  [[ -n ${param_arr[6]} ]]  && cmd+=" --num_warps ${param_arr[6]}"
  [[ -n ${param_arr[7]} ]]  && cmd+=" --num_stages ${param_arr[7]}"
  [[ -n ${param_arr[8]} ]]  && cmd+=" --top_k ${param_arr[8]}"
  [[ -n ${param_arr[9]} ]]  && cmd+=" --compute_type ${param_arr[9]}"
  [[ -n ${param_arr[10]} ]] && cmd+=" --use_fp8_w8a8 ${param_arr[10]}"
  [[ -n ${param_arr[11]} ]] && cmd+=" --use_int8_w8a16 ${param_arr[11]}"
  [[ -n ${param_arr[12]} ]] && cmd+=" --m ${param_arr[12]}"
  [[ -n ${param_arr[13]} ]] && cmd+=" --n ${param_arr[13]}"
  [[ -n ${param_arr[14]} ]] && cmd+=" --k ${param_arr[14]}"
  [[ -n ${param_arr[15]} ]] && cmd+=" --num_experts ${param_arr[15]}"
  [[ -n ${param_arr[16]} ]] && cmd+=" --tune"
  eval $cmd
  if [ $? -ne 0 ]; then
    echo "Error: failed to execute: $cmd"
    exit 1
  fi
done

PARAM_LIST=(
#   0 group_n
#   1 group_k
#   2 BLOCK_SIZE_M
#   3 BLOCK_SIZE_N
#   4 BLOCK_SIZE_K
#   5 GROUP_SIZE_M
#   6 num_warps
#   7 num_stages
#   8 top_k
#   9 compute_type
#   10 use_fp8_w8a8
#   11 use_int8_w8a16
#   12 m
#   13 n
#   14 k
#   15 num_experts
#   16 tune
#  Only DeepSeek-R1/V3 (topk=8) and DeepSeek-V2 (topk=6), BF16 only
#  0    1    2   3    4    5  6  7  8  9     10     11        12  13  14  15  16
  "128, 128, 64, 128, 128, 1, 4, 3, 8, BF16, False, False,    ,   ,   ,    ,   ,"
  "128, 128, 64, 128, 128, 1, 4, 3, 6, BF16, False, False,    ,   ,   ,    ,   ,"
)

for param in "${PARAM_LIST[@]}"
do
  IFS=',' read -ra param_arr <<< "${param// /}"
  cmd="python $DIR/fused_batched_moe.py --output_dir $OUTPUT_DIR"
  [[ -n ${param_arr[0]} ]]  && cmd+=" --group_n ${param_arr[0]}"
  [[ -n ${param_arr[1]} ]]  && cmd+=" --group_k ${param_arr[1]}"
  [[ -n ${param_arr[2]} ]]  && cmd+=" --BLOCK_SIZE_M ${param_arr[2]}"
  [[ -n ${param_arr[3]} ]]  && cmd+=" --BLOCK_SIZE_N ${param_arr[3]}"
  [[ -n ${param_arr[4]} ]]  && cmd+=" --BLOCK_SIZE_K ${param_arr[4]}"
  [[ -n ${param_arr[5]} ]]  && cmd+=" --GROUP_SIZE_M ${param_arr[5]}"
  [[ -n ${param_arr[6]} ]]  && cmd+=" --num_warps ${param_arr[6]}"
  [[ -n ${param_arr[7]} ]]  && cmd+=" --num_stages ${param_arr[7]}"
  [[ -n ${param_arr[8]} ]]  && cmd+=" --top_k ${param_arr[8]}"
  [[ -n ${param_arr[9]} ]]  && cmd+=" --compute_type ${param_arr[9]}"
  [[ -n ${param_arr[10]} ]] && cmd+=" --use_fp8_w8a8 ${param_arr[10]}"
  [[ -n ${param_arr[11]} ]] && cmd+=" --use_int8_w8a16 ${param_arr[11]}"
  [[ -n ${param_arr[12]} ]] && cmd+=" --m ${param_arr[12]}"
  [[ -n ${param_arr[13]} ]] && cmd+=" --n ${param_arr[13]}"
  [[ -n ${param_arr[14]} ]] && cmd+=" --k ${param_arr[14]}"
  [[ -n ${param_arr[15]} ]] && cmd+=" --num_experts ${param_arr[15]}"
  [[ -n ${param_arr[16]} ]] && cmd+=" --tune"
  eval $cmd
  if [ $? -ne 0 ]; then
    echo "Error: failed to execute: $cmd"
    exit 1
  fi
done
