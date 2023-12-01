# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(ExternalProject)
ExternalProject_Add(googletest
  GIT_REPOSITORY    https://git.woa.com/Numerous/LLM/LLM_kernels.git
  GIT_TAG           master
  SOURCE_DIR        "${CMAKE_BINARY_DIR}/llm-kernels-src"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/llm-kernels-build"
  CONFIGURE_COMMAND "cmake -B ./build"
  BUILD_COMMAND     "cd build && make -j"
  INSTALL_COMMAND   "make install"
  TEST_COMMAND      ""
)