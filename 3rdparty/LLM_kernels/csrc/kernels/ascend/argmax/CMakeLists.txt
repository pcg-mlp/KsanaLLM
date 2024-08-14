# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME argmax)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE)

include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_argmax)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_argmax)
execute_process(COMMAND
  /bin/sh ${PROJECT_SOURCE_DIR}/tools/touch_host_stub.sh ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_argmax/host_stub.cpp)

file(GLOB_RECURSE argmax_host_stub_SRCS
  ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_argmax/host_stub.cpp)
message(STATUS "argmax_host_stub_SRCS: ${argmax_host_stub_SRCS}")
add_library(argmax_host_stub ${argmax_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_argmax PUBLIC argmax_host_stub)

add_dependencies(argmax_host_stub llm_kernels_ascend_core_argmax)
add_dependencies(llm_kernels_ascend_argmax llm_kernels_ascend_core_argmax)