# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE BGMV_SRCS *.cu)
list(FILTER BGMV_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_bgmv STATIC ${BGMV_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_bgmv PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_bgmv PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_bgmv PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE BGMV_TEST_SRCS *test.cu)
# copy python script for make test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/bgmv/bgmv_test.py ${CMAKE_CURRENT_BINARY_DIR})
# copy python script for standalone test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/bgmv/bgmv_test.py ${CMAKE_BINARY_DIR})
cc_test(llm_kernels_nvidia_kernel_bgmv_test SRCS ${BGMV_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_bgmv)
