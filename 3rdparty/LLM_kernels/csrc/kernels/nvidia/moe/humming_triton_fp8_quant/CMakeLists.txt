# Copyright 2026 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE TRITON_FP8_QUANT_ALL_FILES
  triton_fp8_quant_creator.py
  generate_triton_fp8_quant.py
  generate_triton_silu_mul_quant.py
)

set(TRITON_FP8_QUANT_MD5_SUM_DIR ${CMAKE_CURRENT_BINARY_DIR}/triton_fp8_quant_md5_sum)
set(TRITON_FP8_QUANT_MD5_FILE ${TRITON_FP8_QUANT_MD5_SUM_DIR}/.triton_fp8_quant.md5sums)

set(TRITON_FP8_QUANT_NEED_REGENERATE TRUE)

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(TRITON_FP8_QUANT_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping triton_fp8_quant kernel generation")
endif()

if(EXISTS ${TRITON_FP8_QUANT_MD5_FILE})
  file(READ ${TRITON_FP8_QUANT_MD5_FILE} PREV_MD5)
  calculate_files_md5("${TRITON_FP8_QUANT_ALL_FILES}" CURRENT_MD5)
  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(TRITON_FP8_QUANT_NEED_REGENERATE FALSE)
    message(STATUS "triton_fp8_quant files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "triton_fp8_quant files MD5 changed, regenerating")
  endif()
endif()

if(TRITON_FP8_QUANT_NEED_REGENERATE)
  message(STATUS "Start generate triton_fp8_quant triton kernel")

  execute_process(COMMAND python triton_fp8_quant_creator.py ${TRITON_KERNEL_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE TRITON_FP8_QUANT_RESULT_VARIABLE
    OUTPUT_VARIABLE TRITON_FP8_QUANT_RESULT_OUTPUT)

  if(TRITON_FP8_QUANT_RESULT_VARIABLE AND NOT TRITON_FP8_QUANT_RESULT_VARIABLE EQUAL 0)
    message(FATAL_ERROR "python triton_fp8_quant_creator.py ${TRITON_KERNEL_PATH} with error:\n${TRITON_FP8_QUANT_RESULT_OUTPUT}")
  endif()

  calculate_files_md5("${TRITON_FP8_QUANT_ALL_FILES}" NEW_MD5)
  file(WRITE ${TRITON_FP8_QUANT_MD5_FILE} ${NEW_MD5})

  message(STATUS "Success generate triton_fp8_quant triton kernel")
endif()