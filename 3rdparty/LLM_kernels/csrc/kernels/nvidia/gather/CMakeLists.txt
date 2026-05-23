# Copyright 2026 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE GATHER_SRCS *.cu)
list(FILTER GATHER_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_gather STATIC ${GATHER_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_gather PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_gather PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_gather PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE GATHER_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_gather_test SRCS ${GATHER_TEST_SRCS} DEPS
  llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_gather)
