# Copyright 2026 Tencent Inc.  All rights reserved.

# Triton kernel; SM90+ only.
if(HAS_SM90 OR HAS_SM90A)
file(GLOB_RECURSE ATTN_RES_ALL_FILES
  attn_res_creator.sh
  generate_attn_res.py
  attn_res_kernel.py
)

set(ATTN_RES_MD5_SUM_DIR ${CMAKE_CURRENT_BINARY_DIR}/attn_res_md5_sum)
set(ATTN_RES_MD5_FILE ${ATTN_RES_MD5_SUM_DIR}/.attn_res.md5sums)

set(ATTN_RES_NEED_REGENERATE TRUE)

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(ATTN_RES_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping attn_res kernel generation")
endif()

if(EXISTS ${ATTN_RES_MD5_FILE})
  file(READ ${ATTN_RES_MD5_FILE} PREV_MD5)
  calculate_files_md5("${ATTN_RES_ALL_FILES}" CURRENT_MD5)
  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(ATTN_RES_NEED_REGENERATE FALSE)
    message(STATUS "attn_res files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "attn_res files MD5 changed, regenerating")
  endif()
endif()

if(ATTN_RES_NEED_REGENERATE)
  message(STATUS "Start generate attn_res triton kernel")

  execute_process(COMMAND bash attn_res_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE ATTN_RES_RESULT_VARIABLE
    OUTPUT_VARIABLE ATTN_RES_RESULT_OUTPUT)

  if(ATTN_RES_RESULT_VARIABLE AND NOT ATTN_RES_RESULT_VARIABLE EQUAL 0)
    message(FATAL_ERROR "bash attn_res_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH} with error:\n${ATTN_RES_RESULT_OUTPUT}")
  endif()

  calculate_files_md5("${ATTN_RES_ALL_FILES}" NEW_MD5)
  file(WRITE ${ATTN_RES_MD5_FILE} ${NEW_MD5})

  message(STATUS "Success generate attn_res triton kernel")
endif()
endif()
