# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

set(nvidia_kernels_LIBS "")
set(nvidia_kernels_SRCS "")

if(WITH_CUDA)
  file(GLOB_RECURSE nvidia_kernels_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/kernels/nvidia/*.cpp)
  list(FILTER nvidia_kernels_SRCS EXCLUDE REGEX ".*test.cpp")

  list(APPEND nvidia_kernels_LIBS -lcudart llm_kernels_nvidia_kernel_paged_attention
    llm_kernels_nvidia_kernel_embedding flash_attn_kernels
    llm_kernels_nvidia_kernel_layernorm llm_kernels_nvidia_kernel_gemm_wrapper
    llm_kernels_nvidia_kernel_add llm_kernels_nvidia_kernel_activation
    llm_kernels_nvidia_kernel_assemble_last_token llm_kernels_nvidia_kernel_cast
    llm_kernels_nvidia_kernel_rotary_embedding llm_kernels_nvidia_kernel_all_reduce
    llm_kernels_nvidia_kernel_permute llm_kernels_nvidia_kernel_alibi
    llm_kernels_nvidia_kernel_greedy)
endif()

message(STATUS "nvidia_kernels_SRCS: ${nvidia_kernels_SRCS}")

set(ascend_kernels_LIBS "")
set(ascend_kernels_SRCS "")

if(WITH_ACL)
  file(GLOB_RECURSE ascend_kernels_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/kernels/ascend/*.cpp)
  list(FILTER ascend_kernels_SRCS EXCLUDE REGEX ".*test.cpp")
  list(APPEND ascend_kernels_LIBS ${ACL_SHARED_LIBS}
    llm_kernels_ascend_pointwise
    llm_kernels_ascend_transpose
    llm_kernels_ascend_permute
    llm_kernels_ascend_embedding
    llm_kernels_ascend_reshape
    llm_kernels_ascend_argmax)
endif()

message(STATUS "ascend_kernels_SRCS: ${ascend_kernels_SRCS}")

add_library(kernels STATIC ${nvidia_kernels_SRCS} ${ascend_kernels_SRCS})
target_link_libraries(kernels PUBLIC utils block_manager ${Python3_LIBRARIES} ${nvidia_kernels_LIBS} ${ascend_kernels_LIBS})
