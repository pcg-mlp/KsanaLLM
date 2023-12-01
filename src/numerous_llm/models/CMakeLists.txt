# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE models_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/models/*.cpp)
list(FILTER models_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "models_SRCS: ${models_SRCS}")

add_library(models STATIC ${models_SRCS})

