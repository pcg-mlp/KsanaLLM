# Copyright 2024 Tencent Inc.  All rights reserved.
get_filename_component(KLLM_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)

set(3RDPARTY_DIR ${KLLM_ROOT_DIR}/3rdparty)

add_subdirectory(utils)
add_subdirectory(kernels)
set(llm_kernels_ascend_libs ${llm_kernels_ascend_libs} PARENT_SCOPE)
