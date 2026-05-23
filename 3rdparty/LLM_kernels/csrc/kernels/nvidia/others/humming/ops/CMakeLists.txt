# Copyright 2026 Tencent Inc. All rights reserved.
#
# Humming ops — top-level dispatch and workspace sizing.

file(GLOB HUMMING_OPS_SRCS *.cu)

add_library(llm_kernels_nvidia_kernel_humming_ops STATIC ${HUMMING_OPS_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_humming_ops PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_humming_ops PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(llm_kernels_nvidia_kernel_humming_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_include_directories(llm_kernels_nvidia_kernel_humming_ops PUBLIC
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(llm_kernels_nvidia_kernel_humming_ops PUBLIC
  -lcuda -lcudart llm_kernels_nvidia_utils
  llm_kernels_nvidia_kernel_humming_jit
  llm_kernels_nvidia_kernel_humming_launcher)
