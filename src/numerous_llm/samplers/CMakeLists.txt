# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE samplers_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/samplers/*.cpp)
list(FILTER samplers_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "samplers_SRCS: ${samplers_SRCS}")

add_library(samplers STATIC ${samplers_SRCS})

