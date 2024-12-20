# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE models_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/base/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/common/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/quant/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/llama/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/qwen/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/qwen2_vl/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/baichuan/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/chatglm/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/gpt/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/common_moe/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/mixtral/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/qwen2_moe/*.cpp)

list(FILTER models_SRCS EXCLUDE REGEX ".*test.cpp")

set(kernels_nvidia_LIBS, "")
set(kernels_ascend_LIBS, "")

if(WITH_CUDA)
  list(APPEND kernels_nvidia_LIBS llm_kernels_nvidia_kernel_asymmetric_gemm)
endif()

add_library(models STATIC ${models_SRCS})
target_link_libraries(models PUBLIC block_manager layers samplers runtime data_hub "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY} ${kernels_nvidia_LIBS})

# for test
file(GLOB_RECURSE models_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/base/*test.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/llama/*test.cpp)
message(STATUS "models_test_SRCS: ${models_test_SRCS} ")

if(WITH_STANDALONE_TEST)
  if(WITH_VLLM_FLASH_ATTN)
    cpp_test(models_test SRCS ${models_test_SRCS} DEPS models runtime data_hub LIBS "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})
  else()
    cpp_test(models_test SRCS ${models_test_SRCS} DEPS models runtime data_hub)
  endif()
endif()
