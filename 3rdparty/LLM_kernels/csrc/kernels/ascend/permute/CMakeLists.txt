# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME permute)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE DEPS)

include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_permute)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_permute)
execute_process(COMMAND
  touch ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_permute/host_stub.cpp)

file(GLOB_RECURSE permute_host_stub_SRCS
  ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_permute/host_stub.cpp)
message(STATUS "permute_host_stub_SRCS: ${permute_host_stub_SRCS}")
add_library(permute_host_stub ${permute_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_permute PUBLIC permute_host_stub)

add_dependencies(permute_host_stub llm_kernels_ascend_core_permute)
add_dependencies(llm_kernels_ascend_permute llm_kernels_ascend_core_permute)