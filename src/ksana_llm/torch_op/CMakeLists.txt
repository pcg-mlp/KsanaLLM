# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE torch_op_SRCS
		 ${PROJECT_SOURCE_DIR}/src/ksana_llm/torch_op/*.cpp)
list(FILTER torch_op_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "torch_op_SRCS: ${torch_op_SRCS}")

add_library(torch_serving SHARED ${torch_op_SRCS})
target_link_libraries(torch_serving PRIVATE "${TORCH_LIBRARIES}" ksana_llm_libs)

