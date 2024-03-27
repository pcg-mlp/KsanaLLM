# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE block_manager_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/block_manager/*.cpp)
list(FILTER block_manager_SRCS EXCLUDE REGEX ".*test.cpp")

if(NOT WITH_CUDA)
  list(FILTER block_manager_SRCS EXCLUDE REGEX ".*nvidia_allocator.*")
endif()

if(NOT WITH_ACL)
  list(FILTER block_manager_SRCS EXCLUDE REGEX ".*ascend_allocator.*")
endif()

message(STATUS "block_manager_SRCS: ${block_manager_SRCS}")

add_library(block_manager STATIC ${block_manager_SRCS})
target_link_libraries(block_manager PUBLIC utils)

# unittest
file(GLOB_RECURSE block_manager_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/block_manager/*test.cpp)

if(NOT WITH_CUDA)
  list(FILTER block_manager_test_SRCS EXCLUDE REGEX ".*nvidia_allocator.*")
endif()

if(NOT WITH_ACL)
  list(FILTER block_manager_test_SRCS EXCLUDE REGEX ".*ascend_allocator.*")
endif()

message(STATUS "block_manager_test_SRCS: ${block_manager_test_SRCS}")
cpp_test(block_manager_test SRCS ${block_manager_test_SRCS} DEPS block_manager runtime)
