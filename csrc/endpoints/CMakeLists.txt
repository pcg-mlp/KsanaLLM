# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)
option(WITH_TRPC_ENDPOINT "Build with trpc endpoint" OFF)

file(GLOB_RECURSE endpoints_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/endpoints/*.cpp)
# 排除 wrapper 目录下所有文件
list(FILTER endpoints_SRCS EXCLUDE REGEX ".*/wrapper/.*")
list(FILTER endpoints_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "endpoints_SRCS: ${endpoints_SRCS}")

add_library(endpoints STATIC ${endpoints_SRCS})
target_link_libraries(endpoints PUBLIC pd_v2_public pd_v2_wire utils)

# 按需引入 trpc 子目录
if(WITH_TRPC_ENDPOINT)
    add_subdirectory(wrapper/trpc)
endif()

# pd_v2 prefill endpoint/completion unit tests (CPU-only, no GPU required).
cpp_test(pd_v2_prefill_completion_test
    SRCS ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2/pd_v2_prefill_completion_test.cpp
    DEPS endpoints)

cpp_test(pd_v2_prefill_endpoint_test
    SRCS ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2/pd_v2_prefill_endpoint_test.cpp
    DEPS endpoints)
