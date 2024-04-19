# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE GREEDY_SRCS *.cu)
list(FILTER GREEDY_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_greedy STATIC ${GREEDY_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_greedy PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_greedy PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_greedy PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE GREEDY_TEST_SRCS *test.cu)
message(STATUS "GREEDY_TEST_SRCS : ${GREEDY_TEST_SRCS}")
cc_test(llm_kernels_nvidia_kernel_greedy_test SRCS ${GREEDY_TEST_SRCS} DEPS
  llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_greedy)
