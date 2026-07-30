# Copyright 2026 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE TOPK_SOFTPLUS_SQRT_SRCS *.cu)
list(FILTER TOPK_SOFTPLUS_SQRT_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_vllm_topk_softplus_sqrt STATIC ${TOPK_SOFTPLUS_SQRT_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_vllm_topk_softplus_sqrt PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_vllm_topk_softplus_sqrt PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_vllm_topk_softplus_sqrt PUBLIC -lcublas -lcudart -lcublasLt ${TORCH_LIBRARIES})
target_link_libraries(llm_kernels_nvidia_kernel_vllm_topk_softplus_sqrt PRIVATE llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE TOPK_SOFTPLUS_SQRT_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_vllm_topk_softplus_sqrt_test SRCS ${TOPK_SOFTPLUS_SQRT_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_vllm_topk_softplus_sqrt)
