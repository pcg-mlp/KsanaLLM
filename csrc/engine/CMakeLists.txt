# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE engine_SRCS
	${PROJECT_SOURCE_DIR}/csrc/engine/*.cpp)
list(FILTER engine_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "engine_SRCS: ${engine_SRCS}")

add_library(engine STATIC ${engine_SRCS})

# for test
file(GLOB_RECURSE engine_test_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/engine/*test.cpp)

message(STATUS "runtime_test_SRCS: ${runtime_test_SRCS}")
cpp_test(engine_test SRCS ${engine_test_SRCS} DEPS engine executor)
