# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE plugins_SRCS
	${PROJECT_SOURCE_DIR}/src/ksana_llm/plugins/*.cpp)
list(FILTER plugins_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "plugins_SRCS: ${plugins_SRCS}")

add_library(plugins STATIC ${plugins_SRCS})
