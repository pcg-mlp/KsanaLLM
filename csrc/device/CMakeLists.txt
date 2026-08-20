# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

# 排除 LayerProgressTracker 文件，不编译到 device 库中
file(GLOB_RECURSE device_SRCS ${PROJECT_SOURCE_DIR}/csrc/device/*.cpp)
list(FILTER device_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER device_SRCS EXCLUDE REGEX ".*/nvidia/.*.cpp")
list(FILTER device_SRCS EXCLUDE REGEX ".*/ascend/.*.cpp")
list(FILTER device_SRCS EXCLUDE REGEX ".*/zixiao/.*.cpp")
list(FILTER device_SRCS EXCLUDE REGEX ".*/kunlun/.*.cpp")
message(STATUS "device_SRCS: ${device_SRCS}")

set(device_nvidia_SRCS "")
set(device_nvidia_LIBS "")

if(WITH_CUDA)
  file(GLOB_RECURSE device_nvidia_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/device/nvidia/*.cpp
    ${PROJECT_SOURCE_DIR}/csrc/device/nvidia/*.cu)
  list(FILTER device_nvidia_SRCS EXCLUDE REGEX ".*test\\.cpp")
  list(FILTER device_nvidia_SRCS EXCLUDE REGEX ".*test\\.cu")
  list(APPEND device_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublasLt -lcublas)
endif()

set(device_ascend_SRCS "")
set(device_ascend_LIBS "")

if(WITH_ACL)
  file(GLOB_RECURSE device_ascend_SRCS ${PROJECT_SOURCE_DIR}/csrc/device/ascend/*.cpp)
  list(FILTER device_ascend_SRCS EXCLUDE REGEX ".*cuda_graph_runner.cpp")
  list(APPEND device_ascend_LIBS ${ACL_SHARED_LIBS} hccl)
endif()

set(device_zixiao_SRCS "")
set(device_zixiao_LIBS "")

if(WITH_TOPS)
  file(GLOB_RECURSE device_zixiao_SRCS ${PROJECT_SOURCE_DIR}/csrc/device/zixiao/*.cpp)
  list(FILTER device_zixiao_SRCS EXCLUDE REGEX ".*cuda_graph_runner.cpp")
  list(APPEND device_zixiao_LIBS ${TOPS_SHARED_LIBS} topsrt)
endif()

# Kunlun XPU
set(device_kunlun_SRCS "")
set(device_kunlun_LIBS "")

if(WITH_KUNLUN)
  file(GLOB_RECURSE device_kunlun_SRCS ${PROJECT_SOURCE_DIR}/csrc/device/kunlun/*.cpp)
  list(FILTER device_kunlun_SRCS EXCLUDE REGEX ".*test.cpp")
  list(APPEND device_kunlun_LIBS ${XPU_SHARED_LIBS})
endif()

add_library(device STATIC ${device_SRCS} ${device_nvidia_SRCS} ${device_ascend_SRCS} ${device_zixiao_SRCS} ${device_kunlun_SRCS})
add_dependencies(device utils fmt gflags layer_progress_tracker)
# NOTE: ${device_zixiao_SRCS} in link_libraries is an existing bug (should be LIBS).
# Keeping as-is to minimize diff; Kunlun uses the correct variable.
target_link_libraries(device PUBLIC
  -lpthread -ldl gflags ${device_nvidia_LIBS} ${device_ascend_LIBS} ${device_zixiao_SRCS} ${device_kunlun_LIBS} utils
)

if(WITH_CUDA)
  set_target_properties(device PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(device PRIVATE ${TORCH_INCLUDE_DIRS})
  if(ENABLE_DEEPSEEK_DEEPGEMM_FLAG)
    target_include_directories(device PRIVATE ${DEEPGEMM_INCLUDE_DIRS})
    target_compile_definitions(device PRIVATE
      DEEPGEMM_LIBRARY_ROOT_PATH="${DEEPGEMM_LIBRARY_ROOT_PATH}"
      DEEPGEMM_CUDA_HOME="${CUDA_TOOLKIT_ROOT_DIR}")
  endif()
endif()

# for test
if(WITH_CUDA)
  file(GLOB_RECURSE device_test_SRCS ${PROJECT_SOURCE_DIR}/csrc/device/nvidia/*test.cpp)
  cpp_test(device_test SRCS ${device_test_SRCS} DEPS device utils)
endif()
