# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE profiler_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/profiler/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/profiler/*.cpp)

list(FILTER profiler_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "profiler_SRCS: ${profiler_SRCS}")

add_library(profiler STATIC ${profiler_SRCS})
target_link_libraries(profiler PUBLIC utils)

# for test
file(GLOB_RECURSE profiler_test_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/profiler/*test.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/profiler/*test.cpp)
message(STATUS "profiler_test_SRCS: ${profiler_test_SRCS}")
cc_test(profiler_test SRCS ${profiler_test_SRCS} DEPS profiler utils)
