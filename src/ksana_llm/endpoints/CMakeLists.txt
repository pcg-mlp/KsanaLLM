# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE endpoints_SRCS
	${PROJECT_SOURCE_DIR}/src/ksana_llm/endpoints/*.cpp)
list(FILTER endpoints_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "endpoints_SRCS: ${endpoints_SRCS}")

add_library(endpoints STATIC ${endpoints_SRCS})
