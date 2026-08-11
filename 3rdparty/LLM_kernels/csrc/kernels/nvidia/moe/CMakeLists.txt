# Copyright 2025 Tencent Inc.  All rights reserved.

add_subdirectory(cutlass_moe)
add_subdirectory(expert_map)
add_subdirectory(moe_wna16)
add_subdirectory(fused_moe_gptq_int4_fp8_kernel)
# hpc-ops FP8 fused MoE. Built unconditionally: the target pins its own
# CUDA_ARCHITECTURES to 90a rather than relying on the SM guard below.
add_subdirectory(hpc_moe)
# For Triton/TileLang kernels,
# generation can be omitted for lower SM architectures.
if(HAS_SM90 OR HAS_SM90A)
  add_subdirectory(fused_moe)
  add_subdirectory(fused_batched_moe)
  add_subdirectory(fused_moe_kernel_gptq_awq)
  add_subdirectory(fused_moe_wfp4afp8)
  add_subdirectory(humming_triton_fp8_quant)
  add_subdirectory(humming_triton_moe_fused_mul_sum)
endif()
