# Copyright 2024 Tencent Inc.  All rights reserved.

# set Nvidia utils target
file(GLOB_RECURSE NVIDIA_UTILS_SRCS *.cu)
add_library(llm_kernels_nvidia_utils STATIC ${NVIDIA_UTILS_SRCS})
target_link_libraries(llm_kernels_nvidia_utils PUBLIC -lcublas -lcudart -lcurand)
set_property(TARGET llm_kernels_nvidia_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
