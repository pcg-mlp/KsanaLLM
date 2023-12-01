# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE runtime_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/runtime/*.cpp)
list(FILTER runtime_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "runtime_SRCS: ${runtime_SRCS}")

add_library(runtime STATIC ${runtime_SRCS})

