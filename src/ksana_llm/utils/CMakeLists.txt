# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE utils_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/*.cpp)
list(FILTER utils_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "utils_SRCS: ${utils_SRCS}")

set(utils_nvidia_SRCS, "")
file(GLOB_RECURSE utils_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/nvidia/*.cpp)
list(FILTER utils_SRCS EXCLUDE REGEX ".*test.cpp")

add_library(utils STATIC ${utils_SRCS} ${utils_nvidia_SRCS})
add_dependencies(utils fmt gflags)
target_link_libraries(utils PUBLIC 
  -lpthread -ldl gflags loguru -lcudart block_manager ksana_llm_yaml ${NCCL_LIBRARIES}
)
