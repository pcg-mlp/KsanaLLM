# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(numerous_llm LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "-std=c++17 -ggdb -O2 -g -Werror=return-type -Wall -Wno-strict-aliasing -Wno-pointer-arith -Wno-ignored-attributes -Wno-deprecated -finline-functions -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-unknown-pragmas")

add_subdirectory(3rdparty)
add_subdirectory(benchmarks)
add_subdirectory(examples)
add_subdirectory(src)
add_subdirectory(tests)
