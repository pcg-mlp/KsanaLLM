# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME embedding)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE)

include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_embedding)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_embedding)
execute_process(COMMAND 
  /bin/sh ${PROJECT_SOURCE_DIR}/tools/touch_host_stub.sh ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_embedding/host_stub.cpp)

file(GLOB_RECURSE embedding_host_stub_SRCS
	${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_embedding/host_stub.cpp)
message(STATUS "embedding_host_stub_SRCS: ${embedding_host_stub_SRCS}")
add_library(embedding_host_stub ${embedding_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_embedding PUBLIC embedding_host_stub)

add_dependencies(embedding_host_stub  llm_kernels_ascend_core_embedding)
add_dependencies(llm_kernels_ascend_embedding llm_kernels_ascend_core_embedding)