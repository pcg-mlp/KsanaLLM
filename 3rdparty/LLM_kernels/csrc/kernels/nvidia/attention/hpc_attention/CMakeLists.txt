# Copyright 2026 Tencent Inc.  All rights reserved.
#
# hpc-ops attention kernels (decode + prefill).
# Source: https://github.com/Tencent/hpc-ops (submodule pinned to 8d300e1)
# Compiled with hpc's own cutlass 4.4.2 (PRIVATE include, isolated from KsanaLLM's cutlass).
# ==============================================================================

set(HPC_OPS_ROOT "${PROJECT_SOURCE_DIR}/3rdparty/hpc-ops")

set(HPC_ATTN_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc_decode_impl.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc_paged_prefill_impl.cu
  ${HPC_OPS_ROOT}/src/attention/decode/smallm_splitk_dim128.cu
  ${HPC_OPS_ROOT}/src/attention/prefill/prefill.cc
  ${HPC_OPS_ROOT}/src/attention/prefill/warp_spec_dim128.cu
  ${HPC_OPS_ROOT}/src/attention/prefill/multi_stage_dim128.cu
  ${HPC_OPS_ROOT}/src/attention/prefill/warp_spec_with_kvcache_dim128.cu
  ${HPC_OPS_ROOT}/src/attention/prefill/multi_stage_with_kvcache_dim128.cu
  ${HPC_OPS_ROOT}/src/attention/prefill/warp_spec_with_kvcache_fp8_dim128.cu
  ${HPC_OPS_ROOT}/src/utils/utils.cc
)

add_library(llm_kernels_nvidia_kernel_hpc_attention STATIC ${HPC_ATTN_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_hpc_attention PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_hpc_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(llm_kernels_nvidia_kernel_hpc_attention PROPERTIES
  CUDA_ARCHITECTURES "90a"
  CXX_STANDARD 17
  CUDA_STANDARD 17
)

target_include_directories(llm_kernels_nvidia_kernel_hpc_attention PRIVATE
  ${HPC_OPS_ROOT}
  ${HPC_OPS_ROOT}/3rd/cutlass/include
)

target_compile_options(llm_kernels_nvidia_kernel_hpc_attention PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --expt-relaxed-constexpr
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

target_link_libraries(llm_kernels_nvidia_kernel_hpc_attention PUBLIC
  -lcuda -lcudart llm_kernels_nvidia_utils)
