# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME layernorm DEPS_KERNELS pointwise elementwise)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE)
