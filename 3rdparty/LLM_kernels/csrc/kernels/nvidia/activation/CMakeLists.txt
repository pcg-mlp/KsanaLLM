# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ACTIVATION_SRCS *.cu)
list(FILTER ACTIVATION_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_activation STATIC ${ACTIVATION_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_activation PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_activation PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_activation PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE ACTIVATION_TEST_SRCS *test.cu)
# copy python script for make test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/activation/activation_test.py ${CMAKE_CURRENT_BINARY_DIR})
# copy python script for standalone test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/activation/activation_test.py ${CMAKE_BINARY_DIR})
cc_test(llm_kernels_nvidia_kernel_activation_test SRCS ${ACTIVATION_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_activation)