#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${KSANA_TRITON_KERNEL_PATH:-}" ]]; then
  OUT="${KSANA_TRITON_KERNEL_PATH}/triton_kernel_files"
else
  OUT="${ROOT}/triton_kernel_files"
fi

run_one() {
  local name="$1"
  local rel="$2"
  local script="$3"
  local dir="${ROOT}/3rdparty/LLM_kernels/csrc/kernels/nvidia/${rel}"
  if [[ ! -d "$dir" ]]; then
    echo "missing kernel dir: $dir" >&2
    exit 1
  fi
  mkdir -p "${OUT}/${name}"
  echo "generate ${name} -> ${OUT}/${name}"
  (cd "$dir" && bash "$script" "$dir" "${OUT}/${name}")
}

run_one fused_moe moe/fused_moe fused_moe_creator.sh
run_one fused_moe_kernel_gptq_awq moe/fused_moe_kernel_gptq_awq fused_moe_kernel_gptq_awq_creator.sh
run_one fused_moe_gptq_int4_fp8_kernel moe/fused_moe_gptq_int4_fp8_kernel fused_moe_gptq_int4_fp8_kernel_creator.sh
run_one kda kda kda_creator.sh
