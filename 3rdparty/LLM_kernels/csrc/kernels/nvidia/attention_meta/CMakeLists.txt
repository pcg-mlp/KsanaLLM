# Copyright 2026 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ATTENTION_META_SRCS *.cu)
list(FILTER ATTENTION_META_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_attention_meta STATIC ${ATTENTION_META_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_attention_meta PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_attention_meta PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_attention_meta PUBLIC -lcudart)

# for test
file(GLOB_RECURSE ATTENTION_META_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_attention_meta_test SRCS ${ATTENTION_META_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_attention_meta)
