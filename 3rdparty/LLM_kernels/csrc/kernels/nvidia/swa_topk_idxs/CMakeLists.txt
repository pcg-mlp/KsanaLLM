# Copyright 2026 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE SWA_TOPK_IDXS_SRCS *.cu)
list(FILTER SWA_TOPK_IDXS_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_swa_topk_idxs STATIC ${SWA_TOPK_IDXS_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_swa_topk_idxs PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_swa_topk_idxs PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_swa_topk_idxs PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE SWA_TOPK_IDXS_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_swa_topk_idxs_test SRCS ${SWA_TOPK_IDXS_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_swa_topk_idxs)
