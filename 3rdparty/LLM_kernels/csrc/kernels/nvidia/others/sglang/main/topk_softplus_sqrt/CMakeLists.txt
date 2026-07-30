# Copyright 2026 Tencent Inc. All rights reserved.
#
# Build rules for the DeepSeek V4 sqrtsoftplus + hash MoE gate kernels
# ported from sglang:
#   python/sglang/jit_kernel/csrc/deepseek_v4/hash_topk.cuh
#   python/sglang/jit_kernel/csrc/moe/moe_fused_gate.cuh

file(GLOB_RECURSE SGLANG_TSS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
list(FILTER SGLANG_TSS_SRCS EXCLUDE REGEX ".*test\\.cu$")
message(STATUS "SGLANG_TSS_SRCS: ${SGLANG_TSS_SRCS}")

add_library(llm_kernels_nvidia_kernel_sglang_topk_softplus_sqrt STATIC ${SGLANG_TSS_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_sglang_topk_softplus_sqrt PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_sglang_topk_softplus_sqrt PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(llm_kernels_nvidia_kernel_sglang_topk_softplus_sqrt PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/others/sglang/main/include)
target_link_libraries(llm_kernels_nvidia_kernel_sglang_topk_softplus_sqrt PUBLIC -lcublas -lcudart -lcublasLt
  llm_kernels_nvidia_utils)

# for test — test source is added in Task 6; guard registration so empty glob does not error.
file(GLOB_RECURSE SGLANG_TSS_TEST_SRCS *test.cu)
if(SGLANG_TSS_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_sglang_topk_softplus_sqrt_test SRCS ${SGLANG_TSS_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_sglang_topk_softplus_sqrt
    llm_kernels_nvidia_kernel_vllm_topk_softplus_sqrt)
endif()
