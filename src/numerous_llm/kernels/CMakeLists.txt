# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE kernels_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/kernels/*.cpp)
list(FILTER kernels_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "kernels_SRCS: ${kernels_SRCS}")

add_library(kernels STATIC ${kernels_SRCS})
target_link_libraries(kernels PUBLIC -lcudart)
