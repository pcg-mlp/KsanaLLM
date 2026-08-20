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
# pd_v2 executor-side bootstrap dlopen's libpd_v2_connector_<backend>.so;
# pd_v2_public + pd_v2_decode (for the PdV2ConnectorInterface header) are
# the engine connector linkable parts; libdl provides dlopen/dlsym.
target_link_libraries(executor PUBLIC layer_progress_tracker pd_v2_public pd_v2_layer_event_tracker ${CMAKE_DL_LIBS})

# executor block manager test
cpp_test(executor_block_manager_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/executor/executor_block_manager_test.cpp
  DEPS executor cache_manager utils)

set(EXECUTOR_TEST_DEPS executor batch_scheduler runtime device utils
    ipc distributed connector transfer engine
    models model_loader cache_manager samplers configure service supervisor)
cpp_test(global_cache_put_worker_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/executor/global_cache_put_worker_test.cpp
  DEPS ${EXECUTOR_TEST_DEPS})

cpp_test(inflight_resource_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/executor/inflight_resource_test.cpp
  DEPS ${EXECUTOR_TEST_DEPS})

set(CUPTI_LIB_DIR "" CACHE PATH "Directory containing libcupti.so.11.8 (nvidia cuda_cupti conda package)")
if(WITH_KUNLUN AND NOT DEFINED _CUPTI_LIB_DIR)
  find_file(_CUPTI_LIB_FILE libcupti.so.11.8
    HINTS $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
          ${CUPTI_LIB_DIR}
    NO_DEFAULT_PATH)
  if(_CUPTI_LIB_FILE)
    get_filename_component(_CUPTI_LIB_DIR "${_CUPTI_LIB_FILE}" DIRECTORY)
  else()
    message(WARNING "libcupti.so.11.8 not found; Kunlun executor test link may fail. "
                    "Set -DCUPTI_LIB_DIR=<path> or install nvidia-cuda-cupti-cu11 in conda.")
  endif()
endif()
if(WITH_KUNLUN AND WITH_TESTING)
  target_link_directories(global_cache_put_worker_test PRIVATE ${_CUPTI_LIB_DIR})
endif()
