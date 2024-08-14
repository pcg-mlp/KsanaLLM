# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME slice)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE DEPS)
include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_slice)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_slice)
execute_process(COMMAND
  /bin/sh ${PROJECT_SOURCE_DIR}/tools/touch_host_stub.sh ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_slice/host_stub.cpp)

file(GLOB_RECURSE slice_host_stub_SRCS
  ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_slice/host_stub.cpp)
message(STATUS "slice_host_stub_SRCS: ${slice_host_stub_SRCS}")
add_library(slice_host_stub ${slice_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_slice PUBLIC slice_host_stub)

add_dependencies(slice_host_stub llm_kernels_ascend_core_slice)
add_dependencies(llm_kernels_ascend_slice llm_kernels_ascend_core_slice)
