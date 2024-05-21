# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE samplers_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/samplers/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/samplers/base/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/samplers/beam_search/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/samplers/topk/*.cpp)
list(FILTER samplers_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "samplers_SRCS: ${samplers_SRCS}")

set(samplers_nvidia_LIBS "")

if(WITH_CUDA)
  list(APPEND samplers_nvidia_LIBS cudart llm_kernels_nvidia_kernel_greedy)
endif()

add_library(samplers STATIC ${samplers_SRCS})
target_link_libraries(samplers PUBLIC block_manager utils ${samplers_nvidia_LIBS})
