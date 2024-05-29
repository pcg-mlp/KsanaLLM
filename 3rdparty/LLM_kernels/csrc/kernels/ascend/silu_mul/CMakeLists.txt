# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME silu_mul)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE)

include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_silu_mul)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_silu_mul)
execute_process(COMMAND 
  touch ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_silu_mul/host_stub.cpp)

file(GLOB_RECURSE silu_mul_host_stub_SRCS
	${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_silu_mul/host_stub.cpp)
message(STATUS "silu_mul_host_stub_SRCS: ${silu_mul_host_stub_SRCS}")
add_library(silu_mul_host_stub ${silu_mul_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_silu_mul PUBLIC silu_mul_host_stub)

add_dependencies(silu_mul_host_stub  llm_kernels_ascend_core_silu_mul)
add_dependencies(llm_kernels_ascend_silu_mul llm_kernels_ascend_core_silu_mul)