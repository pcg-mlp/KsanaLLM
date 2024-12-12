# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE runtime_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/*.cpp)
list(FILTER runtime_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "runtime_SRCS: ${runtime_SRCS}")

set(runtime_nvidia_LIBS, "")

if(WITH_CUDA)
  list(APPEND runtime_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublasLt -lcublas)
endif()

set(runtime_ascend_LIBS, "")

if(WITH_ACL)
  list(APPEND runtime_ascend_LIBS ${ACL_SHARED_LIBS})
  list(FILTER runtime_SRCS EXCLUDE REGEX ".*cuda_graph_runner.cpp")
endif()

set(runtime_zixiao_LIBS, "")

if(WITH_TOPS)
  list(APPEND runtime_ascend_LIBS ${TOPS_SHARED_LIBS})
  list(FILTER runtime_SRCS EXCLUDE REGEX ".*cuda_graph_runner.cpp")
endif()

add_library(runtime STATIC ${runtime_SRCS})
add_dependencies(runtime utils fmt gflags)
target_link_libraries(runtime PUBLIC
  -lpthread -ldl gflags ${runtime_nvidia_LIBS} ${runtime_ascend_LIBS}
  models utils
)

# for test
file(GLOB_RECURSE runtime_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/*test.cpp)
message(STATUS "runtime_test_SRCS: ${runtime_test_SRCS}")
cpp_test(runtime_test SRCS ${runtime_test_SRCS} DEPS runtime)
