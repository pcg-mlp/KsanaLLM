# Copyright 2026 Tencent Inc. All rights reserved.
#
# Build rules for the C4 / C128 KV-compression kernels ported from SGLang DeepSeek-V4:
#   https://github.com/sgl-project/sglang/tree/main/python/sglang/jit_kernel/csrc/deepseek_v4

# The kernels themselves live in header-only .cuh files. We additionally
# compile compressor_wrapper.cu which provides a non-templated C++ entry
# point so that .cpp callers do not need to touch CUDA syntax.

file(GLOB_RECURSE DEEPSEEK_V4_SRCS *.cu)
list(FILTER DEEPSEEK_V4_SRCS EXCLUDE REGEX ".*test\\.cu$")

add_library(llm_kernels_nvidia_kernel_sglang_deepseek_v4 STATIC ${DEEPSEEK_V4_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_sglang_deepseek_v4 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_sglang_deepseek_v4 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(llm_kernels_nvidia_kernel_sglang_deepseek_v4 PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/others/sglang/main/include
)
target_link_libraries(llm_kernels_nvidia_kernel_sglang_deepseek_v4 PUBLIC -lcudart llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE DEEPSEEK_V4_TEST_SRCS *test.cu)
if(DEEPSEEK_V4_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_sglang_deepseek_v4_test SRCS ${DEEPSEEK_V4_TEST_SRCS} DEPS llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_sglang_deepseek_v4)
endif()
