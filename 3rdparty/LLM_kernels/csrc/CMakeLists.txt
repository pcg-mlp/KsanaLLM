# Copyright 2024 Tencent Inc.  All rights reserved.

add_subdirectory(utils)
add_subdirectory(kernels)
set(llm_kernels_ascend_libs ${llm_kernels_ascend_libs} PARENT_SCOPE)
