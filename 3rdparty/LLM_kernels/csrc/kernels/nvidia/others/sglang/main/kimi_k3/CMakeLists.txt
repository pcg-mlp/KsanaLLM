# Copyright 2026 Tencent Inc. All rights reserved.

file(GLOB_RECURSE KIMI_K3_SRCS *.cu)
list(FILTER KIMI_K3_SRCS EXCLUDE REGEX ".*test\\.cu$")

add_library(llm_kernels_nvidia_kernel_sglang_kimi_k3 STATIC ${KIMI_K3_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_sglang_kimi_k3 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_sglang_kimi_k3 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(llm_kernels_nvidia_kernel_sglang_kimi_k3 PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/others/sglang/main/include
)
target_link_libraries(llm_kernels_nvidia_kernel_sglang_kimi_k3 PUBLIC -lcudart llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE KIMI_K3_TEST_SRCS *test.cu)
if(KIMI_K3_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_sglang_kimi_k3_test SRCS ${KIMI_K3_TEST_SRCS} DEPS llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_sglang_kimi_k3)
endif()
