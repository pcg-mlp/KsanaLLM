# Copyright 2026 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE DSPARK_SRCS *.cu)
list(FILTER DSPARK_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_dspark STATIC ${DSPARK_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_dspark PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_dspark PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_dspark PUBLIC
    -lcudart
    llm_kernels_nvidia_kernel_vllm_deepseek_v4)
target_link_libraries(llm_kernels_nvidia_kernel_dspark PRIVATE llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE DSPARK_TEST_SRCS *test.cu)
if(DSPARK_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_dspark_test SRCS ${DSPARK_TEST_SRCS} DEPS
      llm_kernels_nvidia_utils
      llm_kernels_nvidia_kernel_dspark)
endif()
