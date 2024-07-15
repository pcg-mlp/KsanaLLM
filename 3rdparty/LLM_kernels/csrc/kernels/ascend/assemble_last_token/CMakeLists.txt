# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME assemble_last_token)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE DEPS)

include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_assemble_last_token)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_assemble_last_token)
execute_process(COMMAND
  /bin/sh ${PROJECT_SOURCE_DIR}/tools/touch_host_stub.sh ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_assemble_last_token/host_stub.cpp)

file(GLOB_RECURSE assemble_last_token_host_stub_SRCS
  ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_assemble_last_token/host_stub.cpp)
message(STATUS "assemble_last_token_host_stub_SRCS: ${assemble_last_token_host_stub_SRCS}")
add_library(assemble_last_token_host_stub ${assemble_last_token_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_assemble_last_token PUBLIC assemble_last_token_host_stub)

add_dependencies(assemble_last_token_host_stub llm_kernels_ascend_core_assemble_last_token)
add_dependencies(llm_kernels_ascend_assemble_last_token llm_kernels_ascend_core_assemble_last_token)
