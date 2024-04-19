# Copyright 2024 Tencent Inc.  All rights reserved.

if(NOT WITH_TESTING)
  return()
endif()

if(NOT WITH_ACL)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_subdirectory(kernels)
