# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE distributed_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/*.cpp)

list(FILTER distributed_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "distributed_SRCS: ${distributed_SRCS}")

add_library(distributed STATIC ${distributed_SRCS})
target_link_libraries(distributed PUBLIC utils data_hub)

# raw_socket_test
file(GLOB_RECURSE raw_socket_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/raw_socket_test.cpp)

message(STATUS "raw_socket_test_SRCS: ${raw_socket_test_SRCS}")
cpp_test(distributed_raw_socket_test SRCS ${raw_socket_test_SRCS} DEPS utils runtime data_hub LIBS "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})

# control_channel_test
file(GLOB_RECURSE control_channel_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/control_channel_test.cpp)

message(STATUS "control_channel_test_SRCS: ${control_channel_test_SRCS}")
cpp_test(distributed_control_channel_test SRCS ${control_channel_test_SRCS} DEPS utils runtime data_hub LIBS "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})

# data_channel_test
file(GLOB_RECURSE data_channel_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/data_channel_test.cpp)

message(STATUS "data_channel_test_SRCS: ${data_channel_test_SRCS}")
cpp_test(distributed_data_channel_test SRCS ${data_channel_test_SRCS} DEPS utils runtime data_hub LIBS "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})

# distributed_coordinator_test
file(GLOB_RECURSE distributed_coordinator_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/distributed_coordinator_test.cpp)

message(STATUS "distributed_coordinator_test_SRCS: ${distributed_coordinator_test_SRCS}")
cpp_test(distributed_coordinator_test SRCS ${distributed_coordinator_test_SRCS} DEPS utils runtime data_hub LIBS "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})
