# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ADD_SRCS *.cu)
list(FILTER ADD_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_add STATIC ${ADD_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_add PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_add PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_add PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE ADD_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_add_test SRCS ${ADD_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_add)