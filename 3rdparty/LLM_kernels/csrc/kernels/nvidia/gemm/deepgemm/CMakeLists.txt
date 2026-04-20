# Copyright 2025 Tencent Inc.  All rights reserved.

if ((HAS_SM90 OR HAS_SM90A) AND ${CUDA_VERSION} VERSION_GREATER_EQUAL "12.8")
    set(ENABLE_DEEPGEMM 1)
else()
    set(ENABLE_DEEPGEMM 0)
    message(WARNING "SM does not include 90/90a or CUDA version < 12.8, set ENABLE_DEEPGEMM=0")
endif()

# set kernels target
file(GLOB_RECURSE MOE_SRCS *.cu)
list(FILTER MOE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_gemm_deepgemm STATIC ${MOE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_gemm_deepgemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_gemm_deepgemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_definitions(llm_kernels_nvidia_kernel_gemm_deepgemm PUBLIC ENABLE_DEEPGEMM=${ENABLE_DEEPGEMM})
target_link_libraries(llm_kernels_nvidia_kernel_gemm_deepgemm PUBLIC -lcublas -lcudart -lcublasLt
    llm_kernels_nvidia_utils)
if (ENABLE_DEEPGEMM)
    target_link_libraries(llm_kernels_nvidia_kernel_gemm_deepgemm PUBLIC llm_kernels_nvidia_kernel_others_trtllm_dev_common)
    # Only compile for SM90/90a
    set(_deepgemm_archs "")
    if(HAS_SM90)
      list(APPEND _deepgemm_archs 90)
    endif()
    if(HAS_SM90A)
      list(APPEND _deepgemm_archs 90a)
    endif()
    set_target_properties(llm_kernels_nvidia_kernel_gemm_deepgemm PROPERTIES CUDA_ARCHITECTURES "${_deepgemm_archs}")
endif()
target_link_libraries(llm_kernels_nvidia_kernel_gemm_deepgemm PRIVATE cutlass_c4x)