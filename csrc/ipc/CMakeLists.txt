# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE ipc_SRCS
	${PROJECT_SOURCE_DIR}/csrc/ipc/*.cpp
	${PROJECT_SOURCE_DIR}/csrc/ipc/core/*.cpp
	${PROJECT_SOURCE_DIR}/csrc/ipc/functions/*.cpp)
list(FILTER ipc_SRCS EXCLUDE REGEX ".*test.cpp")

# NCCL-specific IPC functions are CUDA-only (Pipeline Parallel)
if(NOT WITH_CUDA)
  list(FILTER ipc_SRCS EXCLUDE REGEX ".*connect_nccl_data_channel.cpp")
  list(FILTER ipc_SRCS EXCLUDE REGEX ".*nccl_operations.cpp")
endif()
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
if(WITH_KUNLUN AND NOT DEFINED _CUPTI_LIB_DIR)
  find_file(_CUPTI_LIB_FILE libcupti.so.11.8
    HINTS $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
          /data/miniconda3/envs/env-3.10/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
    NO_DEFAULT_PATH)
  if(_CUPTI_LIB_FILE)
    get_filename_component(_CUPTI_LIB_DIR "${_CUPTI_LIB_FILE}" DIRECTORY)
  else()
    set(_CUPTI_LIB_DIR "/data/miniconda3/envs/env-3.10/lib/python3.10/site-packages/nvidia/cuda_cupti/lib")
  endif()
endif()
if(WITH_KUNLUN AND WITH_TESTING)
  target_link_directories(ipc_test PRIVATE ${_CUPTI_LIB_DIR})
  target_link_directories(ipc_wrapper_test PRIVATE ${_CUPTI_LIB_DIR})
endif()
