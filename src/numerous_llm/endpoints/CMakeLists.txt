# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE endpoints_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/endpoints/*.cpp)
list(FILTER endpoints_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "endpoints_SRCS: ${endpoints_SRCS}")

add_library(endpoints STATIC ${endpoints_SRCS})

