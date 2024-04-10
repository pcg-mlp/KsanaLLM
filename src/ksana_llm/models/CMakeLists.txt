# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE models_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/base/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/common/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/llama/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/qwen/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/baichuan/*.cpp)

list(FILTER models_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "models_SRCS: ${models_SRCS}")

add_library(models STATIC ${models_SRCS})
target_link_libraries(models PUBLIC block_manager layers samplers "${TORCH_LIBRARIES}")

# for test
file(GLOB_RECURSE models_test_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/base/*test.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/llama/*test.cpp)
message(STATUS "models_test_SRCS: ${models_test_SRCS}")
cpp_test(models_test SRCS ${models_test_SRCS} DEPS models runtime)
