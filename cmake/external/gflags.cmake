# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG master
)

FetchContent_GetProperties(gflags)

if(NOT gflags_POPULATED)
  FetchContent_Populate(gflags)
  cmake_policy(SET CMP0069 NEW)
  add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR})
endif()