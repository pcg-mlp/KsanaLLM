# Copyright 2026 Tencent Inc. All rights reserved.
#
# Marlin MoE WNA16 kernel migrated from SGLang:
#   https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/moe/marlin_moe_wna16

file(GLOB_RECURSE MARLIN_MOE_WNA16_SRCS *.cu)
list(FILTER MARLIN_MOE_WNA16_SRCS EXCLUDE REGEX ".*test\\.cu$")

add_library(llm_kernels_nvidia_kernel_sglang_marlin_moe_wna16 STATIC ${MARLIN_MOE_WNA16_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_sglang_marlin_moe_wna16 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_sglang_marlin_moe_wna16 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(llm_kernels_nvidia_kernel_sglang_marlin_moe_wna16 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_include_directories(llm_kernels_nvidia_kernel_sglang_marlin_moe_wna16 PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(llm_kernels_nvidia_kernel_sglang_marlin_moe_wna16 PUBLIC -lcudart llm_kernels_nvidia_utils)

# Unit tests
file(GLOB_RECURSE MARLIN_MOE_WNA16_TEST_SRCS *test.cu)
if(MARLIN_MOE_WNA16_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_sglang_marlin_moe_wna16_test
    SRCS ${MARLIN_MOE_WNA16_TEST_SRCS}
    DEPS llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_sglang_marlin_moe_wna16)
endif()
