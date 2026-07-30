# Copyright 2026 Tencent Inc. All rights reserved.

file(GLOB_RECURSE DEEPSEEK_V32_SRCS *.cu)
list(FILTER DEEPSEEK_V32_SRCS EXCLUDE REGEX ".*test\\.cu$")

add_library(llm_kernels_nvidia_kernel_sglang_deepseek_v32 STATIC ${DEEPSEEK_V32_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_sglang_deepseek_v32 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_sglang_deepseek_v32 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(llm_kernels_nvidia_kernel_sglang_deepseek_v32 PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/others/sglang/main/include
)
target_link_libraries(llm_kernels_nvidia_kernel_sglang_deepseek_v32 PUBLIC -lcudart llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE DEEPSEEK_V32_TEST_SRCS *test.cu)
if(DEEPSEEK_V32_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_sglang_deepseek_v32_test SRCS ${DEEPSEEK_V32_TEST_SRCS} DEPS llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_sglang_deepseek_v32)
endif()
