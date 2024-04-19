# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE CAST_SRCS *.cu)
list(FILTER CAST_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_cast STATIC ${CAST_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_cast PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_cast PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_cast PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE CAST_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_cast_test SRCS ${CAST_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_cast)