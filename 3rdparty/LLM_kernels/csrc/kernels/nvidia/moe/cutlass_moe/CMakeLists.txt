# Copyright 2025 Tencent Inc.  All rights reserved.

if ((HAS_SM90 OR HAS_SM90A) AND ${CUDA_VERSION} VERSION_GREATER_EQUAL "12.8")
    set(ENABLE_CUTLASSMOE 1)
else()
    set(ENABLE_CUTLASSMOE 0)
    message(WARNING "SM does not include 90/90a or CUDA version < 12.8, set ENABLE_CUTLASSMOE=0")
endif()

# set kernels target
file(GLOB_RECURSE CUTLASS_MOE_SRCS *.cu *.cpp)
list(FILTER CUTLASS_MOE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_moe_cutlass_moe STATIC ${CUTLASS_MOE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_moe_cutlass_moe PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_moe_cutlass_moe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_definitions(llm_kernels_nvidia_kernel_moe_cutlass_moe PUBLIC ENABLE_CUTLASSMOE=${ENABLE_CUTLASSMOE})
target_compile_definitions(llm_kernels_nvidia_kernel_moe_cutlass_moe PRIVATE NDEBUG)
target_compile_definitions(llm_kernels_nvidia_kernel_moe_cutlass_moe PRIVATE ENABLE_BF16)
target_compile_definitions(llm_kernels_nvidia_kernel_moe_cutlass_moe PRIVATE ENABLE_FP4)
target_link_libraries(llm_kernels_nvidia_kernel_moe_cutlass_moe PUBLIC -lcublas -lcudart -lcublasLt
                      llm_kernels_nvidia_utils)
if (ENABLE_CUTLASSMOE)
    target_link_libraries(llm_kernels_nvidia_kernel_moe_cutlass_moe PUBLIC llm_kernels_nvidia_kernel_others_trtllm_dev_thop)
    # Only compile for SM90/90a
    set(_cutlass_moe_archs "")
    if(HAS_SM90)
      list(APPEND _cutlass_moe_archs 90)
    endif()
    if(HAS_SM90A)
      list(APPEND _cutlass_moe_archs 90a)
    endif()
    set_target_properties(llm_kernels_nvidia_kernel_moe_cutlass_moe PROPERTIES CUDA_ARCHITECTURES "${_cutlass_moe_archs}")
endif()
target_link_libraries(llm_kernels_nvidia_kernel_moe_cutlass_moe PRIVATE cutlass_c4x)