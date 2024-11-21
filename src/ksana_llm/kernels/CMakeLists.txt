# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

set(nvidia_kernels_LIBS "")
set(nvidia_kernels_SRCS "")
set(nvidia_kernels_test_SRCS "")

if(WITH_CUDA)
  file(GLOB_RECURSE nvidia_kernels_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/kernels/nvidia/*.cpp)
  list(FILTER nvidia_kernels_SRCS EXCLUDE REGEX ".*test.cpp")

  list(APPEND nvidia_kernels_LIBS -lcudart llm_kernels_nvidia_kernel_paged_attention
    llm_kernels_nvidia_kernel_embedding flash_attn_kernels
    llm_kernels_nvidia_kernel_layernorm llm_kernels_nvidia_kernel_gemm_wrapper
    llm_kernels_nvidia_kernel_add llm_kernels_nvidia_kernel_activation
    llm_kernels_nvidia_kernel_assemble_last_token llm_kernels_nvidia_kernel_cast
    llm_kernels_nvidia_kernel_rotary_embedding llm_kernels_nvidia_kernel_all_reduce
    llm_kernels_nvidia_kernel_permute llm_kernels_nvidia_kernel_alibi
    llm_kernels_nvidia_kernel_greedy llm_kernels_nvidia_kernel_asymmetric_gemm
    llm_kernels_nvidia_kernel_gptq_marlin llm_kernels_nvidia_kernel_weight_only_batched_gemv
    llm_kernels_nvidia_kernel_mixture_of_experts llm_kernels_nvidia_utils)

  file(GLOB_RECURSE nvidia_kernels_test_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/kernels/nvidia/*test.cpp)
endif()

message(STATUS "nvidia_kernels_SRCS: ${nvidia_kernels_SRCS}")

set(ascend_kernels_LIBS "")
set(ascend_kernels_SRCS "")

if(WITH_ACL)
  file(GLOB_RECURSE ascend_kernels_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/kernels/ascend/*.cpp)
  list(FILTER ascend_kernels_SRCS EXCLUDE REGEX ".*test.cpp")
  list(APPEND ascend_kernels_LIBS ${ACL_SHARED_LIBS} atb_plugin_operations)
endif()

message(STATUS "ascend_kernels_SRCS: ${ascend_kernels_SRCS}")

add_library(kernels STATIC ${nvidia_kernels_SRCS} ${ascend_kernels_SRCS})
target_link_libraries(kernels PUBLIC utils block_manager ${Python3_LIBRARIES} ${nvidia_kernels_LIBS} ${ascend_kernels_LIBS})

if(WITH_CUDA)
  set(KERNELS_TEST_PREPARE_CODE ${PROJECT_SOURCE_DIR}/tests/triton_wrapper_test.py)
  execute_process(COMMAND ${PYTHON_PATH} ${KERNELS_TEST_PREPARE_CODE} RESULT_VARIABLE _PYTHON_SUCCESS OUTPUT_VARIABLE RESULT)

  if(NOT _PYTHON_SUCCESS EQUAL 0)
    message(FATAL_ERROR "run python ${PROJECT_SOURCE_DIR}/tests/triton_wrapper_test.py failed.")
  endif()

  execute_process(COMMAND cp ${PROJECT_SOURCE_DIR}/tests/triton_wrapper_test.py "/tmp/" RESULT_VARIABLE _CP_SUCCESS OUTPUT_VARIABLE RESULT)

  if(NOT _CP_SUCCESS EQUAL 0)
    message(FATAL_ERROR "run cp ${PROJECT_SOURCE_DIR}/tests/triton_wrapper_test.py to /tmp/ failed.")
  endif()

  message(STATUS "nvidia_kernels_test_SRCS: ${nvidia_kernels_test_SRCS}")
  cpp_test(kernels_test SRCS ${nvidia_kernels_test_SRCS} DEPS kernels)
endif()