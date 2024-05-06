# Copyright 2024 Tencent Inc.  All rights reserved.

add_ascend_kernel_lib(NAME matmul)
set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs}" PARENT_SCOPE DEPS ${ASCEND_PATH}/devlib/linux/x86_64/libregister.so)
