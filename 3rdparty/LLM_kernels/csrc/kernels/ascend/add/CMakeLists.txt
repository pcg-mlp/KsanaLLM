# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME add)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE)

include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_add)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_add)
execute_process(COMMAND
  /bin/sh ${PROJECT_SOURCE_DIR}/tools/touch_host_stub.sh ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_add/host_stub.cpp)

file(GLOB_RECURSE add_host_stub_SRCS
  ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_add/host_stub.cpp)
message(STATUS "add_host_stub_SRCS: ${add_host_stub_SRCS}")
add_library(add_host_stub ${add_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_add PUBLIC add_host_stub)

add_dependencies(add_host_stub llm_kernels_ascend_core_add)
add_dependencies(llm_kernels_ascend_add llm_kernels_ascend_core_add)
