# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME rmsnorm)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE)

include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_rmsnorm)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_rmsnorm)
execute_process(COMMAND
  /bin/sh ${PROJECT_SOURCE_DIR}/tools/touch_host_stub.sh ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_rmsnorm/host_stub.cpp)

file(GLOB_RECURSE rmsnorm_host_stub_SRCS
  ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_rmsnorm/host_stub.cpp)
message(STATUS "rmsnorm_host_stub_SRCS: ${rmsnorm_host_stub_SRCS}")
add_library(rmsnorm_host_stub ${rmsnorm_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_rmsnorm PUBLIC rmsnorm_host_stub)

add_dependencies(rmsnorm_host_stub llm_kernels_ascend_core_rmsnorm)
add_dependencies(llm_kernels_ascend_rmsnorm llm_kernels_ascend_core_rmsnorm)
