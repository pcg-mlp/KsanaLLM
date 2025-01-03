# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE SAMPLERS_SRCS *.cu)
list(FILTER SAMPLERS_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_samplers STATIC ${SAMPLERS_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_samplers PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_samplers PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_samplers PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE SAMPLERS_TEST_SRCS *test.cu)
message(STATUS "SAMPLERS_TEST_SRCS : ${SAMPLERS_TEST_SRCS}")
cc_test(llm_kernels_nvidia_kernel_samplers_test SRCS ${SAMPLERS_TEST_SRCS} DEPS
  llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_samplers)
