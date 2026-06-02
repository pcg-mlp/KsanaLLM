# Copyright 2026 Tencent Inc. All rights reserved.
#
# Humming MoE GEMM operator with load-time JIT compilation.
# Source: https://github.com/inclusionAI/humming

if(HAS_SM90 OR HAS_SM90A)
  add_subdirectory(jit)
  add_subdirectory(launcher)
  add_subdirectory(ops)
  add_subdirectory(preprocess)

  # Unified interface target — consumers only need to link this one target
  add_library(llm_kernels_nvidia_kernel_humming INTERFACE)
  target_link_libraries(llm_kernels_nvidia_kernel_humming INTERFACE
      llm_kernels_nvidia_kernel_humming_jit
      llm_kernels_nvidia_kernel_humming_launcher
      llm_kernels_nvidia_kernel_humming_ops
      llm_kernels_nvidia_kernel_humming_preprocess
  )
else()
  # Stub library for non-SM90 platforms — provides no-op implementations
  # so that callers link successfully but any runtime use logs an error.
  add_library(llm_kernels_nvidia_kernel_humming_stubs STATIC humming_stubs.cpp)
  set_property(TARGET llm_kernels_nvidia_kernel_humming_stubs PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_include_directories(llm_kernels_nvidia_kernel_humming_stubs PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
  target_link_libraries(llm_kernels_nvidia_kernel_humming_stubs PUBLIC
    -lcuda -lcudart llm_kernels_nvidia_utils)

  add_library(llm_kernels_nvidia_kernel_humming INTERFACE)
  target_link_libraries(llm_kernels_nvidia_kernel_humming INTERFACE
      llm_kernels_nvidia_kernel_humming_stubs
  )
endif()
