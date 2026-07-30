# Copyright 2026 Tencent Inc. All rights reserved.
#
# Build rules for the K3 route_radix MoE gate kernel ported from sglang:
#   python/sglang/kernels/jit/csrc/moe/route_radix.cuh (PR #32541)
#
# The kernel is specialized for 896 experts / top-16 (Kimi K3 only).

file(GLOB_RECURSE ROUTE_RADIX_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
list(FILTER ROUTE_RADIX_SRCS EXCLUDE REGEX ".*test\\.cu$")
message(STATUS "ROUTE_RADIX_SRCS: ${ROUTE_RADIX_SRCS}")

add_library(llm_kernels_nvidia_kernel_sglang_route_radix STATIC ${ROUTE_RADIX_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_sglang_route_radix PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_sglang_route_radix PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(llm_kernels_nvidia_kernel_sglang_route_radix PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/others/sglang/main/include)
target_link_libraries(llm_kernels_nvidia_kernel_sglang_route_radix PUBLIC -lcublas -lcudart -lcublasLt
  llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE ROUTE_RADIX_TEST_SRCS *test.cu)
if(ROUTE_RADIX_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_sglang_route_radix_test SRCS ${ROUTE_RADIX_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_sglang_route_radix
    llm_kernels_nvidia_kernel_vllm_single_group_topk)
endif()
