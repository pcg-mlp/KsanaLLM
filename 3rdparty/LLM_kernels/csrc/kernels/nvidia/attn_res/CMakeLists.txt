# Copyright 2026 Tencent Inc.  All rights reserved.

# Triton kernel; SM90+ only.
if(HAS_SM90 OR HAS_SM90A)
file(GLOB_RECURSE ATTN_RES_ALL_FILES
  attn_res_creator.sh
  generate_attn_res.py
  attn_res_kernel.py
)

ksana_triton_kernel_md5_file(attn_res ATTN_RES_MD5_FILE)

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
  message(STATUS "Queue generate attn_res triton kernel")
  calculate_files_md5("${ATTN_RES_ALL_FILES}" NEW_MD5)
  ksana_enqueue_triton_gen(
    NAME attn_res
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    MD5_FILE ${ATTN_RES_MD5_FILE}
    MD5_VALUE ${NEW_MD5}
    COMMAND bash attn_res_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH})
endif()
endif()
