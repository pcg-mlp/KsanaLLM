# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# prepare cutlass
message(STATUS "Running submodule build to fetch LLM_kernels")

add_subdirectory(3rdparty/LLM_kernels)
include_directories(3rdparty/LLM_kernels)