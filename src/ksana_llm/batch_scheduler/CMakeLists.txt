# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE batch_scheduler_SRCS
	${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_scheduler/*.cpp)
list(FILTER batch_scheduler_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "batch_scheduler_SRCS: ${batch_scheduler_SRCS}")

add_library(batch_scheduler STATIC ${batch_scheduler_SRCS})


cpp_test(batch_scheduler_helper_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_scheduler/batch_scheduler_test_helper_test.cpp DEPS batch_scheduler runtime)
cpp_test(batch_scheduler_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_scheduler/batch_scheduler_test.cpp DEPS batch_scheduler runtime)

