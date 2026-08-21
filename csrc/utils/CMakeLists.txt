# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE utils_SRCS ${PROJECT_SOURCE_DIR}/csrc/utils/*.cpp)
list(FILTER utils_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER utils_SRCS EXCLUDE REGEX ".*/nvidia/.*.cpp")
message(STATUS "utils_SRCS: ${utils_SRCS}")

set(utils_nvidia_SRCS "")
set(utils_nvidia_LIBS "")

if(WITH_CUDA)
  file(GLOB_RECURSE utils_nvidia_SRCS ${PROJECT_SOURCE_DIR}/csrc/utils/nvidia/*.cpp)
  list(APPEND utils_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt -lcuda llm_kernels_nvidia_utils)
endif()

add_library(utils STATIC ${utils_SRCS} ${utils_nvidia_SRCS})
add_dependencies(utils fmt gflags)
# NOTE(fangtang, kunlun-adapt): Boost::dll is gated by WITH_BOOST so that
# configure-only runs (without boost downloaded) can still complete.
set(_utils_boost_libs "")
if(WITH_BOOST)
  set(_utils_boost_libs Boost::dll)
endif()
target_link_libraries(utils PUBLIC
  -lpthread -ldl gflags loguru yaml-cpp profiler ${utils_nvidia_LIBS}
  absl::base absl::strings re2 ${_utils_boost_libs} ${TORCH_LIBRARIES} ${PYTHON_LIBRARY} xgrammar device
)

# for test
if(WITH_STANDALONE_TEST)
  file(GLOB_RECURSE utils_test_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/utils/*test.cpp
    ${PROJECT_SOURCE_DIR}/tests/test.cpp)
  list(FILTER utils_test_SRCS EXCLUDE REGEX ".*environment_test.cpp")
  list(FILTER utils_test_SRCS EXCLUDE REGEX ".*grammar_test.cpp")
  list(FILTER utils_test_SRCS EXCLUDE REGEX ".*forward_test.cpp")
  list(FILTER utils_test_SRCS EXCLUDE REGEX ".*tokenizer_test.cpp")
  message(STATUS "utils_test_SRCS: ${utils_test_SRCS}")
  cpp_test(utils_test SRCS ${utils_test_SRCS} DEPS executor utils)
  cpp_test(grammar_test SRCS ${PROJECT_SOURCE_DIR}/csrc/utils/grammar_test.cpp DEPS utils)
endif()
