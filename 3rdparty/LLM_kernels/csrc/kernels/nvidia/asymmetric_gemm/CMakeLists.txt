# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ASYMMETRIC_GEMM_SRCS *.cc)
list(FILTER ASYMMETRIC_GEMM_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_asymmetric_gemm STATIC ${ASYMMETRIC_GEMM_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_asymmetric_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_asymmetric_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_asymmetric_gemm PUBLIC -lcublas -lcudart -lcublasLt)

add_subdirectory(moe_gemm)
add_subdirectory(int8_gemm)
add_subdirectory(fpA_intB_gemm)
