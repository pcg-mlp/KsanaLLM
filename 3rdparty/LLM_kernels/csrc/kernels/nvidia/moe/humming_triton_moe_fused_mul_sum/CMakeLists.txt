# Copyright 2026 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE TRITON_MOE_FUSED_MUL_SUM_ALL_FILES
  triton_moe_fused_mul_sum_creator.py
  generate_triton_moe_fused_mul_sum.py
)

ksana_triton_kernel_md5_file(triton_moe_fused_mul_sum TRITON_MOE_FUSED_MUL_SUM_MD5_FILE)

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
  message(STATUS "Queue generate triton_moe_fused_mul_sum triton kernel")
  calculate_files_md5("${TRITON_MOE_FUSED_MUL_SUM_ALL_FILES}" NEW_MD5)
  ksana_enqueue_triton_gen(
    NAME triton_moe_fused_mul_sum
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    MD5_FILE ${TRITON_MOE_FUSED_MUL_SUM_MD5_FILE}
    MD5_VALUE ${NEW_MD5}
    COMMAND python triton_moe_fused_mul_sum_creator.py ${TRITON_KERNEL_PATH})
endif()
