# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME cast)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE)

include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_cast)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_cast)
execute_process(COMMAND 
  touch ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_cast/host_stub.cpp)

file(GLOB_RECURSE cast_host_stub_SRCS
	${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_cast/host_stub.cpp)
message(STATUS "cast_host_stub_SRCS: ${cast_host_stub_SRCS}")
add_library(cast_host_stub ${cast_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_cast PUBLIC cast_host_stub)

add_dependencies(cast_host_stub  llm_kernels_ascend_core_cast)
add_dependencies(llm_kernels_ascend_cast llm_kernels_ascend_core_cast)