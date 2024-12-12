# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(RE2_INSTALL_DIR ${THIRD_PARTY_PATH}/install/re2)

FetchContent_Declare(re2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG 2024-07-02
  SOURCE_DIR ${RE2_INSTALL_DIR}
)

FetchContent_MakeAvailable(re2)

message(STATUS "Re2 source directory: ${re2_SOURCE_DIR}")
message(STATUS "Re2 binary directory: ${re2_BINARY_DIR}")

include_directories(${re2_SOURCE_DIR})
