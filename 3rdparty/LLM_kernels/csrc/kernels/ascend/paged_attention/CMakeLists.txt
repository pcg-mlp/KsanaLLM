# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME paged_attention
                      DEPS_KERNELS fused_rotary_embedding permute slice)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE DEPS ${ASCEND_PATH}/devlib/linux/x86_64/libregister.so)
include_directories(${PROJECT_BINARY_DIR}/../../include/llm_kernels_ascend_core_paged_attention)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_paged_attention)
execute_process(COMMAND 
  /bin/sh ${PROJECT_SOURCE_DIR}/tools/touch_host_stub.sh ${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_paged_attention/host_stub.cpp)

file(GLOB_RECURSE paged_attention_host_stub_SRCS
	${PROJECT_BINARY_DIR}/../../auto_gen/llm_kernels_ascend_core_paged_attention/host_stub.cpp)
message(STATUS "paged_attention_host_stub_SRCS: ${paged_attention_host_stub_SRCS}")
add_library(paged_attention_host_stub ${paged_attention_host_stub_SRCS})
target_link_libraries(llm_kernels_ascend_paged_attention PUBLIC paged_attention_host_stub)

add_dependencies(paged_attention_host_stub  llm_kernels_ascend_core_paged_attention)
add_dependencies(llm_kernels_ascend_paged_attention llm_kernels_ascend_core_paged_attention)


include(/usr/local/Ascend/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake/function.cmake)
 
ascendc_compile_definitions(llm_kernels_ascend_core_paged_attention PRIVATE 
  $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
  -DHAVE_WORKSPACE
  -DHAVE_TILING
  )

