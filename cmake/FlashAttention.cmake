# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# prepare cutlass
message(STATUS "Running submodule update to fetch FlashAttention")
find_package(Git QUIET)

set(FLAS_ATTN_PYTHON_SO, "")
set(FLAS_ATTN_VERSION, "")
set(FLAS_ATTN_MINOR_VERSION, "")
execute_process(COMMAND python -c "import torch,flash_attn_2_cuda;print(flash_attn_2_cuda.__file__)" OUTPUT_VARIABLE FLAS_ATTN_PYTHON_SO OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import flash_attn;print(flash_attn.__version__)" OUTPUT_VARIABLE FLAS_ATTN_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import flash_attn;print(flash_attn.__version__.split('.')[1])" OUTPUT_VARIABLE FLAS_ATTN_MINOR_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "FLAS_ATTN_PYTHON_SO: ${FLAS_ATTN_PYTHON_SO}")

# if(NOT "${FLAS_ATTN_PYTHON_SO}" STREQUAL "" AND EXISTS ${FLAS_ATTN_PYTHON_SO} AND EXISTS /etc/tlinux-release)
if(NOT "${FLAS_ATTN_PYTHON_SO}" STREQUAL "" AND EXISTS ${FLAS_ATTN_PYTHON_SO})
  add_library(flash_attn_kernels UNKNOWN IMPORTED)
  set_property(TARGET flash_attn_kernels PROPERTY IMPORTED_LOCATION "${FLAS_ATTN_PYTHON_SO}")
  add_definitions("-DENABLE_FLASH_ATTN_2")
  set(ENABLE_FLASH_ATTN_2 TRUE)
  add_definitions("-DENABLE_FLASH_ATTN_MINOR_${FLAS_ATTN_MINOR_VERSION}")
  message(STATUS "using flash attention ${FLAS_ATTN_VERSION} from python")
else()
  set(FLAS_ATTN_PYTHON_SO, "")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/FlashAttention OUTPUT_VARIABLE FlashAttentionoutput
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/FlashAttention failed with ${GIT_SUBMOD_RESULT}, please checkout FlashAttention submodule")
  endif()

  if(NOT TARGET flash_attn_kernels)
    message(STATUS "using flash_attn 1")
    add_subdirectory(3rdparty/FlashAttention/csrc/flash_attn/src/)
    include_directories(3rdparty/FlashAttention/csrc/flash_attn/src/)
  endif()

  set(ENABLE_FLASH_ATTN_2 FALSE)
endif()