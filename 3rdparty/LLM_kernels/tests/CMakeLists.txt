# Copyright 2024 Tencent Inc.  All rights reserved.

if(NOT WITH_TESTING)
  return()
endif()

if(NOT WITH_ACL)
  include(external/gtest)
endif()

add_subdirectory(kernels)
add_subdirectory(references)
