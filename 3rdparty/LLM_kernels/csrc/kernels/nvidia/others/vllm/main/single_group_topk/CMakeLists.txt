# Copyright 2026 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE SINGLE_GROUP_TOPK_SRCS *.cu)
list(FILTER SINGLE_GROUP_TOPK_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_vllm_single_group_topk STATIC ${SINGLE_GROUP_TOPK_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_vllm_single_group_topk PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_vllm_single_group_topk PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_vllm_single_group_topk PUBLIC -lcublas -lcudart -lcublasLt ${TORCH_LIBRARIES})
target_link_libraries(llm_kernels_nvidia_kernel_vllm_single_group_topk PRIVATE llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE SINGLE_GROUP_TOPK_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_vllm_single_group_topk_test SRCS ${SINGLE_GROUP_TOPK_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_vllm_single_group_topk)
