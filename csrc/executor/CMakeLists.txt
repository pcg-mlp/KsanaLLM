# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

# 为 LayerProgressTracker 添加独立的编译选项
add_library(layer_progress_tracker STATIC
  ${PROJECT_SOURCE_DIR}/csrc/executor/layer_progress_tracker.cpp)

target_link_libraries(layer_progress_tracker PUBLIC
  -lpthread utils gflags)

if(WITH_CUDA)
  cpp_test(layer_progress_tracker_test
    SRCS ${PROJECT_SOURCE_DIR}/csrc/executor/layer_progress_tracker_test.cpp
         ${PROJECT_SOURCE_DIR}/csrc/executor/nccl_executor_handler_test.cpp
    DEPS layer_progress_tracker ipc executor runtime device utils distributed connector
         engine models model_loader cache_manager samplers configure service supervisor)
endif()

file(GLOB_RECURSE executor_SRCS
	${PROJECT_SOURCE_DIR}/csrc/executor/*.cpp)
list(FILTER executor_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER executor_SRCS EXCLUDE REGEX ".*layer_progress_tracker.cpp")
message(STATUS "executor_SRCS: ${executor_SRCS}")

add_library(executor STATIC ${executor_SRCS})
add_dependencies(executor layer_progress_tracker)
target_link_libraries(executor PUBLIC layer_progress_tracker)

# executor block manager test
cpp_test(executor_block_manager_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/executor/executor_block_manager_test.cpp
  DEPS executor cache_manager utils)

# for test
# file(GLOB_RECURSE executor_test_SRCS
#   ${PROJECT_SOURCE_DIR}/csrc/executor/*test.cpp)
# 
# message(STATUS "runtime_test_SRCS: ${runtime_test_SRCS}")
# cpp_test(executor_test SRCS ${executor_test_SRCS} DEPS executor)
