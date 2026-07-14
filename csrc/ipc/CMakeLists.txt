# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE ipc_SRCS
	${PROJECT_SOURCE_DIR}/csrc/ipc/*.cpp
	${PROJECT_SOURCE_DIR}/csrc/ipc/core/*.cpp
	${PROJECT_SOURCE_DIR}/csrc/ipc/functions/*.cpp)
list(FILTER ipc_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "ipc_SRCS: ${ipc_SRCS}")

add_library(ipc STATIC ${ipc_SRCS})
target_link_libraries(ipc PUBLIC executor)

# for test
set(IPC_TEST_DEPS ipc runtime device utils executor distributed
    connector engine models model_loader cache_manager samplers configure service supervisor)
file(GLOB_RECURSE ipc_test_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/ipc/core/*test.cpp
  ${PROJECT_SOURCE_DIR}/csrc/ipc/functions/*test.cpp
  ${PROJECT_SOURCE_DIR}/tests/test.cpp)
message(STATUS "ipc_test_SRCS: ${ipc_test_SRCS}")
cpp_test(ipc_test SRCS ${ipc_test_SRCS} DEPS ${IPC_TEST_DEPS})
cpp_test(ipc_wrapper_test SRCS ${PROJECT_SOURCE_DIR}/csrc/ipc/ipc_func_wrapper_test.cpp DEPS ${IPC_TEST_DEPS})
