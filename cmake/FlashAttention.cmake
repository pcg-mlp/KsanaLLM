# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# prepare cutlass
message(STATUS "Running submodule update to fetch FlashAttention")
find_package(Git QUIET)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/FlashAttention OUTPUT_VARIABLE FlashAttentionoutput
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
if(NOT GIT_SUBMOD_RESULT EQUAL "0")
  message(FATAL_ERROR "git submodule update --init 3rdparty/FlashAttention failed with ${GIT_SUBMOD_RESULT}, please checkout FlashAttention submodule")
endif()
if (NOT TARGET flash_attn_kernels)
  add_subdirectory(3rdparty/FlashAttention/csrc/flash_attn/src/)
  include_directories(3rdparty/FlashAttention/csrc/flash_attn/src/)
endif()