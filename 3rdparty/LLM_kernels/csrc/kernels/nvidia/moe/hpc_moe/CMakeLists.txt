# Copyright 2026 Tencent Inc.  All rights reserved.
#
# hpc-ops FP8 fused MoE kernels (blockwise + per-tensor).
# Source: https://github.com/Tencent/hpc-ops (submodule pinned to 8d300e1)
# Compiled with hpc's own cutlass 4.4.2 (PRIVATE include, isolated from KsanaLLM's cutlass).
# Exposes llm_kernels::nvidia::InvokeHpcBlockwiseFp8Moe / InvokeHpcPerTensorFp8Moe (thin bridges
# over hpc::fuse_moe::fuse_moe_{blockwise,pertensor}_fp8_async), consumed by
# csrc/layers/nvidia/moe_layer.cpp and gated at the layer level by the
# `setting.moe_backend.moe_impl` switch.
# Only the non-torch (*_async) sources are compiled; the torch-dependent entry.cc is
# intentionally excluded.
# ==============================================================================

set(HPC_OPS_ROOT "${PROJECT_SOURCE_DIR}/3rdparty/hpc-ops")

set(HPC_MOE_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc_moe_impl.cu
  ${HPC_OPS_ROOT}/src/fuse_moe/fuse_moe.cu
  ${HPC_OPS_ROOT}/src/fuse_moe/count_and_gather.cu
  ${HPC_OPS_ROOT}/src/fuse_moe/count_and_gather_for_blockwise.cu
  ${HPC_OPS_ROOT}/src/fuse_moe/reduce.cu
  ${HPC_OPS_ROOT}/src/group_gemm/group_gemm_blockwise_fp8.cu
  ${HPC_OPS_ROOT}/src/group_gemm/group_gemm_pertensor_fp8.cu
  ${HPC_OPS_ROOT}/src/activation/activation.cu
  ${HPC_OPS_ROOT}/src/utils/utils.cc
)

add_library(llm_kernels_nvidia_kernel_hpc_moe STATIC ${HPC_MOE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_hpc_moe PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_hpc_moe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(llm_kernels_nvidia_kernel_hpc_moe PROPERTIES
  CUDA_ARCHITECTURES "90a"
  CXX_STANDARD 17
  CUDA_STANDARD 17
)

target_include_directories(llm_kernels_nvidia_kernel_hpc_moe PRIVATE
  ${HPC_OPS_ROOT}
  ${HPC_OPS_ROOT}/3rd/cutlass/include
)

target_compile_options(llm_kernels_nvidia_kernel_hpc_moe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -std=c++17
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
    -DNDEBUG
    -DHPC_GIT_HASH_STR="8d300e1"
    -DHPC_VERSION_STR="external"
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wno-unused-result
  >
)

target_link_libraries(llm_kernels_nvidia_kernel_hpc_moe PUBLIC
  -lcuda -lcudart llm_kernels_nvidia_utils)
