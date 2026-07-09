# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE layers_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/layers/cpu/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/layers/*.cpp)
list(FILTER layers_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/nvidia/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/ascend/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/zixiao/.*.cpp")
if(NOT WITH_CUDA)
  list(FILTER layers_SRCS EXCLUDE REGEX ".*/flashinfer_.*\\.cpp")
endif()

message(STATUS "layers_SRCS: ${layers_SRCS}")

set(layers_nvidia_SRCS "")
set(layers_nvidia_LIBS "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_nvidia_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/layers/nvidia/*.cpp
    ${PROJECT_SOURCE_DIR}/csrc/layers/nvidia/*.cu)
  list(FILTER layers_nvidia_SRCS EXCLUDE REGEX ".*test\.cpp")
  list(FILTER layers_nvidia_SRCS EXCLUDE REGEX ".*test\.cu")
  list(APPEND layers_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)
endif()

set(layers_ascend_SRCS "")
set(layers_ascend_LIBS "")

if(WITH_ACL)
  file(GLOB_RECURSE layers_ascend_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/ascend/*.cpp)
  list(APPEND layers_ascend_LIBS ${ACL_SHARED_LIBS} llm_kernels_ascend_utils_common
    atb_attention)
endif()

set(layers_zixiao_SRCS "")
set(layers_zixiao_LIBS "")

if(WITH_TOPS)
  file(GLOB_RECURSE layers_zixiao_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/zixiao/*.cpp)
  list(APPEND layers_zixiao_LIBS ${TOPS_SHARED_LIBS})
endif()

add_library(layers STATIC ${layers_SRCS} ${layers_nvidia_SRCS} ${layers_ascend_SRCS} ${layers_zixiao_SRCS})

target_link_libraries(layers PUBLIC
  kernels ${layers_nvidia_LIBS} ${layers_ascend_LIBS} layer_progress_tracker pd_v2_layer_event_tracker runtime
)

# for test
set(layers_test_nvidia_SRCS "")
set(layers_test_ascend_SRCS "")
set(layers_test_zixiao_SRCS "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_test_nvidia_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/nvidia/*test.cpp)
endif()

if(WITH_ACL)
  file(GLOB_RECURSE layers_test_ascend_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/ascend/*test.cpp)
endif()

set(layers_test_common_DEPS layers runtime utils executor)

cpp_test(matmul_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/matmul_layer_test.cpp
  DEPS ${layers_test_common_DEPS})

cpp_test(moe_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/moe_layer_test.cpp
       ${layers_test_nvidia_SRCS}
  DEPS ${layers_test_common_DEPS})

cpp_test(attention_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/attention_layer_test.cpp
  DEPS ${layers_test_common_DEPS})

cpp_test(misc_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/misc_layer_test.cpp
       ${layers_test_ascend_SRCS}
  DEPS ${layers_test_common_DEPS})

cpp_test(fp4_moe_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/fp4_moe_layer_test.cpp
  DEPS ${layers_test_common_DEPS})

cpp_test(topk_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/nvidia/topk_layer_test.cpp
  DEPS ${layers_test_common_DEPS})
