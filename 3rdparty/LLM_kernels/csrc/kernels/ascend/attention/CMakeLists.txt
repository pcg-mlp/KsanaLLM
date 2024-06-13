# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME attention
  DEPS_KERNELS concat transpose elementwise matmul rotary_embedding permute reshape)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE)
