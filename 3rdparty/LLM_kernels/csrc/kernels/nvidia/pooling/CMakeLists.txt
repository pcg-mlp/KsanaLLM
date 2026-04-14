# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE POOLING_SRCS *.cu)
list(FILTER POOLING_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_pooling STATIC ${POOLING_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_pooling PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_pooling PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_pooling PUBLIC -lcudart)

# for test
file(GLOB_RECURSE POOLING_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_pooling_test SRCS ${POOLING_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_pooling)
