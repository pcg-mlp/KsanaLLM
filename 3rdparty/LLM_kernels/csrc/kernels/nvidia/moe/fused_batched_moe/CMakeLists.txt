# Copyright 2026 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE FUSED_BATCHED_MOE_ALL_FILES
  fused_batched_moe_creator.sh
  fused_batched_moe.py
)

ksana_triton_kernel_md5_file(fused_batched_moe FUSED_BATCHED_MOE_MD5_FILE)
set(FUSED_BATCHED_MOE_NEED_REGENERATE TRUE)

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(FUSED_BATCHED_MOE_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping fused_batched_moe kernel generation")
endif()

if(EXISTS ${FUSED_BATCHED_MOE_MD5_FILE})
  file(READ ${FUSED_BATCHED_MOE_MD5_FILE} PREV_MD5)

  calculate_files_md5("${FUSED_BATCHED_MOE_ALL_FILES}" CURRENT_MD5)

  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(FUSED_BATCHED_MOE_NEED_REGENERATE FALSE)
    message(STATUS "Files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "Files MD5 changed, regenerating")
    message(STATUS "Previous MD5: ${PREV_MD5}")
    message(STATUS "Current MD5: ${CURRENT_MD5}")
  endif()
endif()

if(FUSED_BATCHED_MOE_NEED_REGENERATE)
  message(STATUS "Queue generate fused_batched_moe triton kernel")
  calculate_files_md5("${FUSED_BATCHED_MOE_ALL_FILES}" NEW_MD5)
  ksana_enqueue_triton_gen(
    NAME fused_batched_moe
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    MD5_FILE ${FUSED_BATCHED_MOE_MD5_FILE}
    MD5_VALUE ${NEW_MD5}
    COMMAND bash fused_batched_moe_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH})
endif()

# copy python script for runtime use
execute_process(COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/fused_batched_moe.py ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/fused_batched_moe.py ${CMAKE_BINARY_DIR})
execute_process(COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/../common.py ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/../common.py ${CMAKE_BINARY_DIR})
