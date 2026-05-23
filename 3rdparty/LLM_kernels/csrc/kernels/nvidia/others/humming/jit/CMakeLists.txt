# Copyright 2026 Tencent Inc. All rights reserved.
#
# Humming JIT compiler — C++ only (no CUDA), links against CUDA driver API.

set(HUMMING_JIT_SRCS humming_jit_compiler.cpp)

add_library(llm_kernels_nvidia_kernel_humming_jit STATIC ${HUMMING_JIT_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_humming_jit PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(llm_kernels_nvidia_kernel_humming_jit PUBLIC
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(llm_kernels_nvidia_kernel_humming_jit PUBLIC
  -lcuda llm_kernels_nvidia_utils)
