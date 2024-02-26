# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE runtime_SRCS
		 ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/*.cpp)
list(FILTER runtime_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "runtime_SRCS: ${runtime_SRCS}")

add_library(runtime STATIC ${runtime_SRCS})
add_dependencies(runtime utils fmt gflags)
target_link_libraries(runtime PUBLIC 
  -lpthread -ldl gflags ${NCCL_LIBRARIES} -lcublas -lcublasLt -lcudart
  models utils 
)

# for test
file(GLOB_RECURSE runtime_test_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/*test.cpp)
message(STATUS "runtime_test_SRCS: ${runtime_test_SRCS}")
cpp_test(runtime_test SRCS ${runtime_test_SRCS} DEPS runtime)