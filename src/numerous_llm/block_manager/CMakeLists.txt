# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE block_manager_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/block_manager/*.cpp)
list(FILTER block_manager_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "block_manager_SRCS: ${block_manager_SRCS}")

add_library(block_manager STATIC ${block_manager_SRCS})

# for test
file(GLOB_RECURSE block_manager_test_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/block_manager/*test.cpp)
message(STATUS "block_manager_test_SRCS: ${block_manager_test_SRCS}")
cc_test(block_manager_test SRCS ${block_manager_test_SRCS} DEPS block_manager)
