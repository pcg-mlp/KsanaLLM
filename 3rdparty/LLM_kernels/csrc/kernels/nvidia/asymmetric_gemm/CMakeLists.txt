# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ASYMMETRIC_GEMM_SRCS *.cc *.cu)
list(FILTER ASYMMETRIC_GEMM_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_asymmetric_gemm STATIC ${ASYMMETRIC_GEMM_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_asymmetric_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_asymmetric_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_asymmetric_gemm PUBLIC -lcublas -lcudart -lcublasLt)

add_subdirectory(moe_gemm)
add_subdirectory(int8_gemm)
add_subdirectory(fpA_intB_gemm)
add_subdirectory(weightOnlyBatchedGemv)

# for test
file(GLOB_RECURSE ADD_TEST_SRCS asymmetric_gemm_test.cu)
cc_test(llm_kernels_nvidia_kernel_asymmetric_gemm_test SRCS ${ADD_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_asymmetric_gemm)