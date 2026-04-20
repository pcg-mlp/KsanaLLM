# Copyright 2024 Tencent Inc.All rights reserved.

# Set kernels target
file(GLOB_RECURSE BLOCKWISE_GEMM_SRCS *.cu)
list(FILTER BLOCKWISE_GEMM_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_blockwise_gemm STATIC ${BLOCKWISE_GEMM_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_blockwise_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_blockwise_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_blockwise_gemm PUBLIC -lcublas -lcudart -lcublasLt)
target_link_libraries(llm_kernels_nvidia_kernel_blockwise_gemm PRIVATE cutlass_c4x ${TORCH_LIBRARIES})

# blockwise_gemm only supports SM90/90a
if(HAS_SM90 OR HAS_SM90A)
  add_definitions(-DENABLE_BLOCKWISE_GEMM=ON)
  # auto enable PDL-related instructions
  target_compile_options(llm_kernels_nvidia_kernel_blockwise_gemm PRIVATE "-DCUTLASS_ENABLE_GDC_FOR_SM90=1")
  # Override CUDA_ARCHITECTURES to only compile for SM90/90a
  set(_blockwise_gemm_archs "")
  if(HAS_SM90)
    list(APPEND _blockwise_gemm_archs 90)
  endif()
  if(HAS_SM90A)
    list(APPEND _blockwise_gemm_archs 90a)
  endif()
  set_target_properties(llm_kernels_nvidia_kernel_blockwise_gemm PROPERTIES CUDA_ARCHITECTURES "${_blockwise_gemm_archs}")
endif()

# For test
file(GLOB_RECURSE BLOCKWISE_GEMM_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_blockwise_gemm_test SRCS ${BLOCKWISE_GEMM_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_blockwise_gemm)
