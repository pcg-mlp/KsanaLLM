#!/bin/bash

# 任何一条 python 命令失败立刻终止，避免静默漏产 cubin/json 导致运行时缺文件。
set -euo pipefail

# 多卡并行调度依赖关联数组（declare -A），需要 bash >= 4.0。
# macOS 自带 bash 3.2 跑不动；Linux 训练机一般是 bash 4.x/5.x，无需担心。
if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
  echo "[wfp4afp8_creator][ERROR] requires bash >= 4.0 (current: ${BASH_VERSION})." >&2
  echo "[wfp4afp8_creator][ERROR] On macOS: 'brew install bash' then run with /usr/local/bin/bash." >&2
  exit 1
fi

# ============================================================================
# 用法：
#   bash fused_moe_wfp4afp8_creator.sh <DIR> <OUTPUT_DIR>
#
# 多卡并行（向后兼容，无需改调用方）：
#   - 默认 GPU 池来源（按优先级）：
#       1) 环境变量 PARALLEL_GPUS="0,1,2,3" 显式指定
#       2) CUDA_VISIBLE_DEVICES 中列出的全部 GPU
#       3) nvidia-smi 检测到的全部 GPU
#       4) 退化到单 GPU 0（与旧行为兼容）
#   - 强制串行：PARALLEL_GPUS="0" bash creator.sh ...
#
# 子任务日志：${OUTPUT_DIR}/.creator_logs/<kname>.log
# 24 个候选组合并发分配到 GPU 池上跑，N 卡近似 N 倍提速。
# ============================================================================

DIR=$1
OUTPUT_DIR=$2

mkdir -p "$OUTPUT_DIR"

# 转成绝对路径，避免任何相对路径歧义；后续 verify 用同一份路径 ls。
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
echo "[wfp4afp8_creator] Resolved OUTPUT_DIR=${OUTPUT_DIR}"

# -------- GPU 池解析 --------
parse_gpu_pool() {
  if [[ -n "${PARALLEL_GPUS:-}" ]]; then
    echo "${PARALLEL_GPUS//,/ }"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "${CUDA_VISIBLE_DEVICES//,/ }"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' '
    return
  fi
  echo "0"
}
read -ra GPU_POOL <<< "$(parse_gpu_pool)"
NUM_GPUS=${#GPU_POOL[@]}
if [[ $NUM_GPUS -eq 0 ]]; then
  GPU_POOL=(0)
  NUM_GPUS=1
fi
echo "[wfp4afp8_creator] GPU pool: ${GPU_POOL[*]} (parallelism=${NUM_GPUS})"

# 不让外部 CUDA_VISIBLE_DEVICES 误干扰子任务；每个子任务自己显式设。
unset CUDA_VISIBLE_DEVICES

# -------- 子任务日志目录 --------
LOG_DIR="${OUTPUT_DIR}/.creator_logs"
mkdir -p "$LOG_DIR"

# -------- 候选枚举 --------
# MXFP4 group_size 固定为 32（OCP MX 标准），不再作为外层维度
Candidate_BLOCK_SIZE_M=(16 32 64)
Candidate_Types=("FP16" "BF16")
Candidate_Quant=("True" "False")
# fc1 (up_gate) 的 top_k 取自模型 expert_topk，不同模型不同：
#   - DSv3 / 通用 MoE: top_k=8
#   - DSv4-pro:        top_k=6
# fc2 (down) 固定 top_k=1（在 InvokeFusedMoeKernelFunc 内 hardcoded），不在这里展开。
Candidate_TopK_FC1=(6 8)

# 与 fused_moe_wfp4afp8.py:441-450 dump_kernel 的命名约定保持完全一致。
expected_kernel_name() {
  local BM=$1 CT=$2 CQ=$3 MRW=$4 TK=$5
  echo "fused_moe_wfp4afp8_kernel_BLOCK_SIZE_M_${BM}_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_MUL_ROUTED_WEIGHT_${MRW}_top_k_${TK}_compute_type_${CT}_group_size_32_quant_a_per_tensor_${CQ}"
}

# -------- 单任务 worker：固定 GPU + 独立 log --------
# python 自身退出 0 不代表 dump 一定成功（早期版本里有 try/except 之类），所以显式 stat。
run_one_on_gpu() {
  local GPU=$1 BM=$2 CT=$3 CQ=$4 MRW=$5 TK=$6
  local kname log_path
  kname=$(expected_kernel_name "$BM" "$CT" "$CQ" "$MRW" "$TK")
  log_path="${LOG_DIR}/${kname}.log"
  {
    echo "[$(date '+%H:%M:%S')] gpu=${GPU} >>> ${kname}"
    CUDA_VISIBLE_DEVICES="$GPU" python "$DIR/fused_moe_wfp4afp8.py" \
      --BLOCK_SIZE_M "$BM" --BLOCK_SIZE_N 128 --BLOCK_SIZE_K 128 --GROUP_SIZE_M 1 \
      --MUL_ROUTED_WEIGHT "$MRW" --top_k "$TK" \
      --compute_type "$CT" \
      --group_size 32 \
      --quant_a_per_tensor "$CQ" \
      --m 222 --n 7168 --k 2048 --num_experts 256 \
      --output_dir "$OUTPUT_DIR" --tune
    local cubin_path="${OUTPUT_DIR}/${kname}.cubin"
    local json_path="${OUTPUT_DIR}/${kname}.json"
    if [[ ! -s "$cubin_path" ]]; then
      echo "[ERROR] missing cubin after generation: ${cubin_path}" >&2
      exit 1
    fi
    if [[ ! -s "$json_path" ]]; then
      echo "[ERROR] missing json after generation: ${json_path}" >&2
      exit 1
    fi
    local size
    size=$(stat -c%s "$cubin_path" 2>/dev/null || stat -f%z "$cubin_path")
    echo "[$(date '+%H:%M:%S')] gpu=${GPU} <<< OK ${kname} (${size} bytes)"
  } >"$log_path" 2>&1
}

# -------- 任务列表扁平化 --------
TASKS=()
for BM in "${Candidate_BLOCK_SIZE_M[@]}"; do
  for CT in "${Candidate_Types[@]}"; do
    for CQ in "${Candidate_Quant[@]}"; do
      for TK in "${Candidate_TopK_FC1[@]}"; do
        TASKS+=("${BM}|${CT}|${CQ}|False|${TK}")
      done
      TASKS+=("${BM}|${CT}|${CQ}|True|1")
    done
  done
done
TOTAL=${#TASKS[@]}
echo "[wfp4afp8_creator] Total tasks: ${TOTAL}"

# -------- 调度循环：GPU 池 + polling 收割 --------
# 用 polling (sleep 0.5s) 而非 `wait -n`，避免对 bash 5.1+ 的依赖；编译每个 cubin
# 通常 30~120s，0.5s 轮询粒度的开销可忽略。
declare -A GPU_PID    # GPU_id -> pid（占用中）
declare -A PID_GPU    # pid -> GPU_id
declare -A PID_KNAME  # pid -> kernel_name（用于失败日志定位）

cleanup() {
  if [[ ${#PID_GPU[@]} -gt 0 ]]; then
    echo "[wfp4afp8_creator] cleanup: killing ${#PID_GPU[@]} pending jobs"
    for pid in "${!PID_GPU[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

acquire_idle_gpu() {
  for g in "${GPU_POOL[@]}"; do
    if [[ -z "${GPU_PID[$g]:-}" ]]; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

START_TS=$(date +%s)
TASK_IDX=0
DONE_COUNT=0
FAIL_COUNT=0

while [[ $TASK_IDX -lt $TOTAL || ${#PID_GPU[@]} -gt 0 ]]; do
  # (1) 把任务派到所有空闲 GPU
  while [[ $TASK_IDX -lt $TOTAL ]]; do
    if ! gpu=$(acquire_idle_gpu); then
      break
    fi
    IFS='|' read -r BM CT CQ MRW TK <<< "${TASKS[$TASK_IDX]}"
    kname=$(expected_kernel_name "$BM" "$CT" "$CQ" "$MRW" "$TK")
    run_one_on_gpu "$gpu" "$BM" "$CT" "$CQ" "$MRW" "$TK" &
    pid=$!
    GPU_PID[$gpu]=$pid
    PID_GPU[$pid]=$gpu
    PID_KNAME[$pid]=$kname
    printf "[wfp4afp8_creator] [%2d/%2d] dispatched pid=%d gpu=%s: %s\n" \
      "$((TASK_IDX + 1))" "$TOTAL" "$pid" "$gpu" "$kname"
    TASK_IDX=$((TASK_IDX + 1))
  done

  # (2) 没有 inflight 直接退出
  if [[ ${#PID_GPU[@]} -eq 0 ]]; then
    break
  fi

  # (3) 轮询：找已经完成的子任务，收割 exit code
  sleep 0.5
  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      ec=0
      wait "$pid" || ec=$?
      gpu=${PID_GPU[$pid]}
      kname=${PID_KNAME[$pid]}
      DONE_COUNT=$((DONE_COUNT + 1))
      log_path="${LOG_DIR}/${kname}.log"
      if [[ $ec -eq 0 ]]; then
        printf "[wfp4afp8_creator] [%2d/%2d] OK   pid=%d gpu=%s: %s\n" \
          "$DONE_COUNT" "$TOTAL" "$pid" "$gpu" "$kname"
      else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        printf "[wfp4afp8_creator][ERROR] [%2d/%2d] FAIL pid=%d gpu=%s exit=%d: %s\n" \
          "$DONE_COUNT" "$TOTAL" "$pid" "$gpu" "$ec" "$kname" >&2
        echo "[wfp4afp8_creator][ERROR] log tail (${log_path}):" >&2
        tail -30 "$log_path" >&2 || true
      fi
      unset "GPU_PID[$gpu]"
      unset "PID_GPU[$pid]"
      unset "PID_KNAME[$pid]"
    fi
  done

  # (4) 任一失败：停止派发新任务，等剩余 inflight 完成后 exit 1
  if [[ $FAIL_COUNT -gt 0 ]]; then
    TASK_IDX=$TOTAL
  fi
done

ELAPSED=$(($(date +%s) - START_TS))
echo "[wfp4afp8_creator] Generation phase done in ${ELAPSED}s (${DONE_COUNT} tasks, ${FAIL_COUNT} failures)"

if [[ $FAIL_COUNT -ne 0 ]]; then
  echo "[wfp4afp8_creator][ERROR] ${FAIL_COUNT} task(s) failed, see logs under ${LOG_DIR}" >&2
  exit 1
fi

# -------- 终极校验：明确列出 DSv4-pro 实际会查的 6 份关键文件，缺一不可 --------
echo "[wfp4afp8_creator] Final verification for DSv4-pro critical files:"
required=(
  "fused_moe_wfp4afp8_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_MUL_ROUTED_WEIGHT_False_top_k_6_compute_type_BF16_group_size_32_quant_a_per_tensor_False"
  "fused_moe_wfp4afp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_MUL_ROUTED_WEIGHT_False_top_k_6_compute_type_BF16_group_size_32_quant_a_per_tensor_False"
  "fused_moe_wfp4afp8_kernel_BLOCK_SIZE_M_64_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_MUL_ROUTED_WEIGHT_False_top_k_6_compute_type_BF16_group_size_32_quant_a_per_tensor_False"
  "fused_moe_wfp4afp8_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_MUL_ROUTED_WEIGHT_True_top_k_1_compute_type_BF16_group_size_32_quant_a_per_tensor_False"
  "fused_moe_wfp4afp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_MUL_ROUTED_WEIGHT_True_top_k_1_compute_type_BF16_group_size_32_quant_a_per_tensor_False"
  "fused_moe_wfp4afp8_kernel_BLOCK_SIZE_M_64_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_MUL_ROUTED_WEIGHT_True_top_k_1_compute_type_BF16_group_size_32_quant_a_per_tensor_False"
)
missing=0
for kname in "${required[@]}"; do
  for ext in cubin json; do
    local_path="${OUTPUT_DIR}/${kname}.${ext}"
    if [[ ! -s "$local_path" ]]; then
      echo "[wfp4afp8_creator][ERROR] FINAL missing: $local_path" >&2
      missing=1
    fi
  done
done
if [[ $missing -ne 0 ]]; then
  echo "[wfp4afp8_creator][ERROR] one or more required artifacts missing — see above." >&2
  exit 1
fi
echo "[wfp4afp8_creator] All required DSv4-pro artifacts present under: ${OUTPUT_DIR}"
