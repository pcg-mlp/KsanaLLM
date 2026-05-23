# Copyright 2026 Tencent Inc. All rights reserved.
#
# Humming launcher — TMA descriptors + cuLaunchKernelEx.

file(GLOB HUMMING_LAUNCHER_SRCS *.cu)
list(FILTER HUMMING_LAUNCHER_SRCS EXCLUDE REGEX ".*reference.*")

add_library(llm_kernels_nvidia_kernel_humming_launcher STATIC ${HUMMING_LAUNCHER_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_humming_launcher PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_humming_launcher PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(llm_kernels_nvidia_kernel_humming_launcher PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_include_directories(llm_kernels_nvidia_kernel_humming_launcher PUBLIC
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(llm_kernels_nvidia_kernel_humming_launcher PUBLIC
  -lcuda -lcudart llm_kernels_nvidia_utils)
