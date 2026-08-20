# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE configure_SRCS ${PROJECT_SOURCE_DIR}/csrc/configure/*.cpp)
list(FILTER configure_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "configure_SRCS: ${configure_SRCS}")

add_library(configure STATIC ${configure_SRCS})
add_dependencies(configure fmt gflags)
# NOTE(fangtang, kunlun-adapt): Boost::dll is gated by WITH_BOOST so that
# configure-only runs (without boost downloaded) can still complete.
set(_configure_boost_libs "")
if(WITH_BOOST)
  set(_configure_boost_libs Boost::dll)
endif()
target_link_libraries(configure PUBLIC
  -lpthread -ldl gflags loguru yaml-cpp profiler absl::base absl::strings re2 ${_configure_boost_libs} xgrammar
)

# for test
if(WITH_STANDALONE_TEST)
  # Only include schedule_config_parser_test for now
  # Other configure tests may have additional dependencies
  set(configure_test_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/configure/schedule_config_parser_test.cpp)
  message(STATUS "configure_test_SRCS: ${configure_test_SRCS}")
  
  cpp_test(schedule_config_parser_test SRCS ${configure_test_SRCS} DEPS configure)
  cpp_test(quant_config_parser_test SRCS ${PROJECT_SOURCE_DIR}/csrc/configure/quant_config_parser_test.cpp DEPS configure models runtime executor LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
  cpp_test(model_config_parser_test SRCS ${PROJECT_SOURCE_DIR}/csrc/configure/model_config_parser_test.cpp DEPS configure models runtime executor LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
  
  cpp_test(environment_test SRCS ${PROJECT_SOURCE_DIR}/csrc/configure/environment_test.cpp DEPS configure models runtime executor LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
  # Other tests can be added here with their specific dependencies
endif()
