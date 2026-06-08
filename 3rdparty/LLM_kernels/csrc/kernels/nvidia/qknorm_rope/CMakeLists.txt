# Copyright 2026 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE QKNORM_ROPE_SRCS *.cu)
list(FILTER QKNORM_ROPE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_qknorm_rope STATIC ${QKNORM_ROPE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_qknorm_rope PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_qknorm_rope PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_qknorm_rope PUBLIC -lcudart)

# for test
file(GLOB_RECURSE QKNORM_ROPE_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_qknorm_rope_test SRCS ${QKNORM_ROPE_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_qknorm_rope
    llm_kernels_nvidia_kernel_layernorm
    llm_kernels_nvidia_kernel_rotary_embedding)
