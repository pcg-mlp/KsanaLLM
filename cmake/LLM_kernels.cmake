# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(ExternalProject)
ExternalProject_Add(LLM_kernels
  GIT_REPOSITORY    https://git.woa.com/Numerous/LLM/LLM_kernels.git
  GIT_TAG           master
  SOURCE_DIR        "${PROJECT_SOURCE_DIR}/3rdparty/LLM_kernels"
  BINARY_DIR        "${PROJECT_SOURCE_DIR}/3rdparty/LLM_kernels"
  CMAKE_ARGS        -DSM=${SM}
)

link_directories(
  ${PROJECT_SOURCE_DIR}/3rdparty/LLM_kernels/lib
)