cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(periphery_SRCS,"")
file(GLOB_RECURSE periphery_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/periphery/*.cpp)
list(FILTER periphery_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "periphery_SRCS: ${periphery_SRCS}")

add_library(periphery STATIC ${periphery_SRCS})
