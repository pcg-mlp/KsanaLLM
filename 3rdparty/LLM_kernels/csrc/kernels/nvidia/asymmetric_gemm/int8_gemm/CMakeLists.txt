# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ASYMMETRIC_GEMM_INT8_SRCS *.cu)
list(FILTER ASYMMETRIC_GEMM_INT8_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_asymmetric_gemm_int8 STATIC ${ASYMMETRIC_GEMM_INT8_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_asymmetric_gemm_int8 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_asymmetric_gemm_int8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_asymmetric_gemm_int8 PUBLIC -lcublas -lcudart -lcublasLt 
    llm_kernels_nvidia_kernel_asymmetric_gemm)
