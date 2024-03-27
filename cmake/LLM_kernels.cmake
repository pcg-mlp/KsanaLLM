# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# prepare cutlass
message(STATUS "Running submodule update to fetch LLM_kernels")
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/LLM_kernels
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_SUBMOD_RESULT)

if(NOT GIT_SUBMOD_RESULT EQUAL "0")
  message(FATAL_ERROR "git submodule update --init 3rdparty/LLM_kernels failed with ${GIT_SUBMOD_RESULT}, please checkout LLM_kernels submodule")
endif()

if(NOT TARGET embedding)
  add_subdirectory(3rdparty/LLM_kernels)
  include_directories(3rdparty/LLM_kernels)
endif()