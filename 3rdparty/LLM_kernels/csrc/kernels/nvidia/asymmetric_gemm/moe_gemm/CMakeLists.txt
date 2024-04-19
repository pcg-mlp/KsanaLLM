# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ASYMMETRIC_GEMM_MMOE_SRCS *.cu)
list(FILTER ASYMMETRIC_GEMM_MMOE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_asymmetric_gemm_moe STATIC ${ASYMMETRIC_GEMM_MMOE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_asymmetric_gemm_moe PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_asymmetric_gemm_moe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_asymmetric_gemm_moe PUBLIC -lcublas -lcudart -lcublasLt 
    llm_kernels_nvidia_kernel_asymmetric_gemm)
