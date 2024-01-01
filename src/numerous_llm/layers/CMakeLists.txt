# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE layers_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/layers/*.cpp)
list(FILTER layers_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "layers_SRCS: ${layers_SRCS}")

add_library(layers STATIC ${layers_SRCS})
target_link_libraries(layers PUBLIC kernels llm_kernels_nvidia_kernel_rotary_embedding) 

# for test
file(GLOB_RECURSE layers_test_SRCS
		 ${PROJECT_SOURCE_DIR}/src/numerous_llm/layers/*test.cpp
     ${PROJECT_SOURCE_DIR}/src/numerous_llm/layers/nvidia/*test.cpp)
cc_test(layers_test SRCS ${layers_test_SRCS} DEPS layers runtime)