# Copyright 2023 Tencent Inc.  All rights reserved.

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# set test target
file(GLOB_RECURSE NUMEROUS_LLM_UNITTEST_SRCS *.cu
                                             *.cc
                                             *.cpp)
add_executable(numerous_llm_unittests ${NUMEROUS_LLM_UNITTEST_SRCS})
set_property(TARGET numerous_llm_unittests PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET numerous_llm_unittests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(numerous_llm_unittests PUBLIC gtest_main gmock -lcudart)