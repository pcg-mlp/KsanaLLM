# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE kernels_SRCS
		 ${PROJECT_SOURCE_DIR}/src/ksana_llm/kernels/*.cpp)
list(FILTER kernels_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "kernels_SRCS: ${kernels_SRCS}")

add_library(kernels STATIC ${kernels_SRCS})
target_link_libraries(kernels PUBLIC -lcudart utils block_manager 
	llm_kernels_nvidia_kernel_paged_attention llm_kernels_nvidia_kernel_embedding
        flash_attn_kernels ${Python3_LIBRARIES}
	llm_kernels_nvidia_kernel_layernorm llm_kernels_nvidia_kernel_gemm_wrapper
	llm_kernels_nvidia_kernel_add llm_kernels_nvidia_kernel_activation
	llm_kernels_nvidia_kernel_assemble_last_token llm_kernels_nvidia_kernel_cast
        llm_kernels_nvidia_kernel_rotary_embedding)
