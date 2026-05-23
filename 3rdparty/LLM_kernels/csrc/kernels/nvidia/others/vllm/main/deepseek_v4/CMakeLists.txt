# Copyright 2026 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE DEEPSEEK_V4_SRCS *.cu)
list(FILTER DEEPSEEK_V4_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_vllm_deepseek_v4 STATIC ${DEEPSEEK_V4_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_vllm_deepseek_v4 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_vllm_deepseek_v4 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(llm_kernels_nvidia_kernel_vllm_deepseek_v4 PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/others/sglang/main/include
)
target_link_libraries(llm_kernels_nvidia_kernel_vllm_deepseek_v4 PUBLIC -lcudart)
target_link_libraries(llm_kernels_nvidia_kernel_vllm_deepseek_v4 PRIVATE llm_kernels_nvidia_utils)

file(GLOB_RECURSE DEEPSEEK_V4_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_vllm_deepseek_v4_test SRCS ${DEEPSEEK_V4_TEST_SRCS} DEPS
    llm_kernels_nvidia_kernel_vllm_deepseek_v4)
