# Copyright 2026 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE TRITON_MOE_FUSED_MUL_SUM_ALL_FILES
  triton_moe_fused_mul_sum_creator.py
  generate_triton_moe_fused_mul_sum.py
)

set(TRITON_MOE_FUSED_MUL_SUM_MD5_SUM_DIR ${CMAKE_CURRENT_BINARY_DIR}/triton_moe_fused_mul_sum_md5_sum)
set(TRITON_MOE_FUSED_MUL_SUM_MD5_FILE ${TRITON_MOE_FUSED_MUL_SUM_MD5_SUM_DIR}/.triton_moe_fused_mul_sum.md5sums)

set(TRITON_MOE_FUSED_MUL_SUM_NEED_REGENERATE TRUE)

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(TRITON_MOE_FUSED_MUL_SUM_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping triton_moe_fused_mul_sum kernel generation")
endif()

if(EXISTS ${TRITON_MOE_FUSED_MUL_SUM_MD5_FILE})
  file(READ ${TRITON_MOE_FUSED_MUL_SUM_MD5_FILE} PREV_MD5)
  calculate_files_md5("${TRITON_MOE_FUSED_MUL_SUM_ALL_FILES}" CURRENT_MD5)
  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(TRITON_MOE_FUSED_MUL_SUM_NEED_REGENERATE FALSE)
    message(STATUS "triton_moe_fused_mul_sum files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "triton_moe_fused_mul_sum files MD5 changed, regenerating")
  endif()
endif()

if(TRITON_MOE_FUSED_MUL_SUM_NEED_REGENERATE)
  message(STATUS "Start generate triton_moe_fused_mul_sum triton kernel")

  execute_process(COMMAND python triton_moe_fused_mul_sum_creator.py ${TRITON_KERNEL_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE TRITON_MOE_FUSED_MUL_SUM_RESULT_VARIABLE
    OUTPUT_VARIABLE TRITON_MOE_FUSED_MUL_SUM_RESULT_OUTPUT)

  if(TRITON_MOE_FUSED_MUL_SUM_RESULT_VARIABLE AND NOT TRITON_MOE_FUSED_MUL_SUM_RESULT_VARIABLE EQUAL 0)
    message(FATAL_ERROR "python triton_moe_fused_mul_sum_creator.py ${TRITON_KERNEL_PATH} with error:\n${TRITON_MOE_FUSED_MUL_SUM_RESULT_OUTPUT}")
  endif()

  calculate_files_md5("${TRITON_MOE_FUSED_MUL_SUM_ALL_FILES}" NEW_MD5)
  file(WRITE ${TRITON_MOE_FUSED_MUL_SUM_MD5_FILE} ${NEW_MD5})

  message(STATUS "Success generate triton_moe_fused_mul_sum triton kernel")
endif()
