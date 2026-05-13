# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# prepare cutlass
message(STATUS "Running submodule build to fetch LLM_kernels")

# Suppress all warnings from LLM_kernels compilation
set(CMAKE_CXX_FLAGS_SAVED "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS_SAVED "${CMAKE_C_FLAGS}")
set(CMAKE_CUDA_FLAGS_SAVED "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

add_subdirectory(3rdparty/LLM_kernels)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SAVED}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_SAVED}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_SAVED}")

include_directories(SYSTEM 3rdparty/LLM_kernels)