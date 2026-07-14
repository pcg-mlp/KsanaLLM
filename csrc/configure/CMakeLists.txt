# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE configure_SRCS ${PROJECT_SOURCE_DIR}/csrc/configure/*.cpp)
list(FILTER configure_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "configure_SRCS: ${configure_SRCS}")

add_library(configure STATIC ${configure_SRCS})
add_dependencies(configure fmt gflags)
target_link_libraries(configure PUBLIC
  -lpthread -ldl gflags loguru yaml-cpp profiler absl::base absl::strings re2 Boost::dll xgrammar
)

# for test
if(WITH_STANDALONE_TEST)
  # Only include schedule_config_parser_test for now
  # Other configure tests may have additional dependencies
  set(configure_test_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/configure/schedule_config_parser_test.cpp)
  message(STATUS "configure_test_SRCS: ${configure_test_SRCS}")
  
  cpp_test(schedule_config_parser_test SRCS ${configure_test_SRCS} DEPS configure)
  
  # cpp_test(environment_test SRCS ${PROJECT_SOURCE_DIR}/csrc/configure/environment_test.cpp DEPS configure)
  # Other tests can be added here with their specific dependencies
endif()
