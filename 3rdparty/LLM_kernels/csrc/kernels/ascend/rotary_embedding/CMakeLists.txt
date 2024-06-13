# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME rotary_embedding
                      DEPS_KERNELS concat slice gather pointwise elementwise)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE)
