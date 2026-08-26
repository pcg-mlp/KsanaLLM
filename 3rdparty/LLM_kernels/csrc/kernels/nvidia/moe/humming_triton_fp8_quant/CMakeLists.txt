# Copyright 2026 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE TRITON_FP8_QUANT_ALL_FILES
  triton_fp8_quant_creator.py
  generate_triton_fp8_quant.py
  generate_triton_silu_mul_quant.py
)

ksana_triton_kernel_md5_file(triton_fp8_quant TRITON_FP8_QUANT_MD5_FILE)

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
  message(STATUS "Queue generate triton_fp8_quant triton kernel")
  calculate_files_md5("${TRITON_FP8_QUANT_ALL_FILES}" NEW_MD5)
  ksana_enqueue_triton_gen(
    NAME triton_fp8_quant
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    MD5_FILE ${TRITON_FP8_QUANT_MD5_FILE}
    MD5_VALUE ${NEW_MD5}
    COMMAND python triton_fp8_quant_creator.py ${TRITON_KERNEL_PATH})
endif()