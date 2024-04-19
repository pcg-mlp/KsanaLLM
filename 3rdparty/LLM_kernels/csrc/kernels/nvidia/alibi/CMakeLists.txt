# Copyright 2024 Tencent Inc.  All rights reserved.

# set alibi kernels target
file(GLOB_RECURSE ALIBI_SRCS *.cu)
list(FILTER ALIBI_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_alibi STATIC ${ALIBI_SRCS})
target_link_libraries(llm_kernels_nvidia_kernel_alibi PUBLIC -lcudart llm_kernels_nvidia_utils)
set_property(TARGET llm_kernels_nvidia_kernel_alibi PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_alibi PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# for test
file(GLOB_RECURSE ALIBI_TEST_SRCS *test.cu)
message(STATUS "ALIBI_TEST_SRCS : ${ALIBI_TEST_SRCS}")
cc_test(llm_kernels_nvidia_kernel_alibi_test SRCS ${ALIBI_TEST_SRCS} DEPS
  llm_kernels_nvidia_utils
  llm_kernels_nvidia_kernel_alibi)
