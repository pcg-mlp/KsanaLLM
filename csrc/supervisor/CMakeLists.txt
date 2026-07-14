# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE supervisor_SRCS
	${PROJECT_SOURCE_DIR}/csrc/supervisor/*.cpp)
list(FILTER supervisor_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "supervisor_SRCS: ${supervisor_SRCS}")

add_library(supervisor STATIC ${supervisor_SRCS})

# for test
# file(GLOB_RECURSE supervisor_test_SRCS
#   ${PROJECT_SOURCE_DIR}/csrc/supervisor/*test.cpp)
# 
# message(STATUS "runtime_test_SRCS: ${runtime_test_SRCS}")
# cpp_test(supervisor_test SRCS ${supervisor_test_SRCS} DEPS supervisor)
