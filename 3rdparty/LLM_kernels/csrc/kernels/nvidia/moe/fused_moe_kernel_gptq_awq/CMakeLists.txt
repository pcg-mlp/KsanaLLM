# Copyright 2025 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE FUSED_MOE_KERNEL_GPTQ_AWQ_ALL_FILES
  fused_moe_kernel_gptq_awq_creator.sh
  fused_moe_kernel_gptq_awq.py
)

ksana_triton_kernel_md5_file(fused_moe_kernel_gptq_awq FUSED_MOE_KERNEL_GPTQ_AWQ_MD5_FILE)

set(FUSED_MOE_KERNEL_GPTQ_AWQ_NEED_REGENERATE TRUE)

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(FUSED_MOE_KERNEL_GPTQ_AWQ_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping fused_moe_gptq_awq kernel generation")
endif()

if(EXISTS ${FUSED_MOE_KERNEL_GPTQ_AWQ_MD5_FILE})
  file(READ ${FUSED_MOE_KERNEL_GPTQ_AWQ_MD5_FILE} PREV_MD5)

  calculate_files_md5("${FUSED_MOE_KERNEL_GPTQ_AWQ_ALL_FILES}" CURRENT_MD5)

  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(FUSED_MOE_KERNEL_GPTQ_AWQ_NEED_REGENERATE FALSE)
    message(STATUS "Files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "Files MD5 changed, regenerating")
    message(STATUS "Previous MD5: ${PREV_MD5}")
    message(STATUS "Current MD5: ${CURRENT_MD5}")
  endif()
endif()

if(FUSED_MOE_KERNEL_GPTQ_AWQ_NEED_REGENERATE)
  message(STATUS "Queue generate fused_moe_kernel_gptq_awq triton kernel")
  calculate_files_md5("${FUSED_MOE_KERNEL_GPTQ_AWQ_ALL_FILES}" NEW_MD5)
  ksana_enqueue_triton_gen(
    NAME fused_moe_kernel_gptq_awq
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    MD5_FILE ${FUSED_MOE_KERNEL_GPTQ_AWQ_MD5_FILE}
    MD5_VALUE ${NEW_MD5}
    COMMAND bash fused_moe_kernel_gptq_awq_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH})
endif()
