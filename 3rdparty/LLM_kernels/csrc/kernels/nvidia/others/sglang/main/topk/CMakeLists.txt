# Copyright 2026 Tencent Inc. All rights reserved.
#
# Build rules for the top-K transform kernels ported from SGLang DeepSeek-V4:
#   https://github.com/sgl-project/sglang/tree/main/python/sglang/jit_kernel/csrc/deepseek_v4
#   (topk.cuh, topk_1024.cuh @ commit 4807f6c5220c8ac53e395d6d2ed493cda241bc8e)
#
# The kernels themselves live in header-only .cuh files. We additionally
# compile topk_wrapper.cu which provides a templated C++ entry point so that
# .cpp callers do not need to touch CUDA syntax.

file(GLOB_RECURSE TOPK_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
list(FILTER TOPK_SRCS EXCLUDE REGEX ".*test\\.cu$")
message(STATUS "TOPK_SRCS: ${TOPK_SRCS}")

add_library(llm_kernels_nvidia_kernel_sglang_topk STATIC ${TOPK_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_sglang_topk PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_sglang_topk PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(llm_kernels_nvidia_kernel_sglang_topk PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/others/sglang/main/include)
target_link_libraries(llm_kernels_nvidia_kernel_sglang_topk PUBLIC -lcublas -lcudart -lcublasLt llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE TOPK_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_sglang_topk_test SRCS ${TOPK_TEST_SRCS} DEPS llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_sglang_topk)
