# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE utils_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/*.cpp)
list(FILTER utils_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER utils_SRCS EXCLUDE REGEX ".*/nvidia/.*.cpp")
list(FILTER utils_SRCS EXCLUDE REGEX ".*/ascend/.*.cpp")
message(STATUS "utils_SRCS: ${utils_SRCS}")

set(utils_nvidia_SRCS, "")
set(utils_nvidia_LIBS, "")

if(WITH_CUDA)
  file(GLOB_RECURSE utils_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/nvidia/*.cpp)
  list(APPEND utils_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)
endif()

set(utils_ascend_SRCS, "")
set(utils_ascend_LIBS, "")

if(WITH_ACL)
  file(GLOB_RECURSE utils_ascend_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/ascend/*.cpp)
  list(APPEND utils_ascend_LIBS ${ACL_SHARED_LIBS})
endif()

add_library(utils STATIC ${utils_SRCS} ${utils_nvidia_SRCS} ${utils_ascend_SRCS})
add_dependencies(utils fmt gflags)
target_link_libraries(utils PUBLIC
  -lpthread -ldl gflags loguru block_manager ksana_llm_yaml ${utils_nvidia_LIBS} ${utils_ascend_LIBS}
)

# for test
if(WITH_CUDA AND WITH_STANDALONE_TEST)
  file(GLOB_RECURSE utils_test_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/*test.cpp)
  message(STATUS "utils_test_SRCS: ${utils_test_SRCS}")
  cpp_test(utils_test SRCS ${utils_test_SRCS} DEPS runtime utils)
endif()
