# Copyright 2024 Tencent Inc.  All rights reserved.

# set masked multihead attention kernels target
file(GLOB_RECURSE ROTARY_EMBEDDING_SRCS *.cu)
list(FILTER ROTARY_EMBEDDING_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_rotary_embedding STATIC ${ROTARY_EMBEDDING_SRCS})
target_link_libraries(llm_kernels_nvidia_kernel_rotary_embedding PUBLIC -lcudart llm_kernels_nvidia_utils)
set_property(TARGET llm_kernels_nvidia_kernel_rotary_embedding PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_rotary_embedding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# for test
file(GLOB_RECURSE ROTARY_EMBEDDING_TEST_SRCS *test.cu)
message(STATUS "ROTARY_EMBEDDING_TEST_SRCS : ${ROTARY_EMBEDDING_TEST_SRCS}")

if(WITH_STANDALONE_TEST)
  cc_test(llm_kernels_nvidia_kernel_rotary_embedding_test SRCS ${ROTARY_EMBEDDING_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_rotary_embedding)
endif()
