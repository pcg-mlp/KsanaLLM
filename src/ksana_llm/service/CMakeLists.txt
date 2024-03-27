# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE service_SRCS
	${PROJECT_SOURCE_DIR}/src/ksana_llm/service/*.cpp)
list(FILTER service_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "service_SRCS: ${service_SRCS}")

add_library(service STATIC ${service_SRCS})
