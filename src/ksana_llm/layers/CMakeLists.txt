# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE layers_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/cpu/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/*.cpp)
list(FILTER layers_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/nvidia/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/ascend/.*.cpp")
message(STATUS "layers_SRCS: ${layers_SRCS}")

set(layers_nvidia_SRCS, "")
set(layers_nvidia_LIBS, "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/nvidia/*.cpp)
  list(APPEND layers_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)
endif()

set(layers_ascend_SRCS, "")
set(layers_ascend_LIBS, "")

if(WITH_ACL)
  file(GLOB_RECURSE layers_ascend_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/ascend/*.cpp)
  list(APPEND layers_ascend_LIBS ${ACL_SHARED_LIBS} llm_kernels_ascend_utils_common
    llm_kernels_ascend_embedding
    llm_kernels_ascend_paged_attention)
endif()

add_library(layers STATIC ${layers_SRCS} ${layers_nvidia_SRCS} ${layers_ascend_SRCS})
target_link_libraries(layers PUBLIC kernels ${layers_nvidia_LIBS} ${layers_ascend_LIBS})

# for test
set(layers_test_nvidia_SRCS, "")
set(layers_test_ascend_SRCS, "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_test_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/nvidia/*test.cpp)
endif()

if(WITH_ACL)
  file(GLOB_RECURSE layers_test_ascend_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/ascend/*test.cpp)
  add_library(extra_deps SHARED ${layers_SRCS} ${layers_nvidia_SRCS} ${layers_ascend_SRCS})
endif()

file(GLOB_RECURSE layers_test_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/*test.cpp
  ${layers_test_nvidia_SRCS} ${layers_test_ascend_SRCS})
message(STATUS "layers_test_SRCS: ${layers_test_SRCS}")
cpp_test(layers_test SRCS ${layers_test_SRCS} DEPS layers runtime utils)
