# Copyright 2024 Tencent Inc.  All rights reserved.

# set embedding kernels target
file(GLOB_RECURSE EMBEDDING_SRCS *.cu)
list(FILTER EMBEDDING_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_embedding STATIC ${EMBEDDING_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_embedding PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_embedding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_embedding PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE EMBEDDING_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_embedding_test SRCS ${EMBEDDING_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_embedding)