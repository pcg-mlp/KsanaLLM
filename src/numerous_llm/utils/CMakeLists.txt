# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE utils_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/utils/*.cpp)
list(FILTER utils_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "utils_SRCS: ${utils_SRCS}")

add_library(utils STATIC ${utils_SRCS})

