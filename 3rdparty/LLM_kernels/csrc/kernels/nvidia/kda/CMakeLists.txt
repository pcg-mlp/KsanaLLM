# Copyright 2026 Tencent Inc.  All rights reserved.

# Triton kernels generated for the Kimi-K3 deployment target; SM90+ only.
if(HAS_SM90 OR HAS_SM90A)
# MD5 tracks operator / generator sources only. Colocated *_test.py files must
# not trigger AOT regeneration.
file(GLOB_RECURSE KDA_ALL_FILES
  kda_creator.sh
  generate_kda.py
  *.py
)
list(FILTER KDA_ALL_FILES EXCLUDE REGEX "_test\\.py$")

set(KDA_MD5_SUM_DIR ${CMAKE_CURRENT_BINARY_DIR}/kda_md5_sum)
set(KDA_MD5_FILE ${KDA_MD5_SUM_DIR}/.kda.md5sums)

set(KDA_NEED_REGENERATE TRUE)

# The requested SSM dtypes select which state-kernel artifacts exist, so they
# are part of the cache key: changing KLLM_KDA_SSM_DTYPE must regenerate even
# though no source file changed.
set(KDA_SSM_DTYPE "float32 bfloat16")
if(DEFINED ENV{KLLM_KDA_SSM_DTYPE})
  set(KDA_SSM_DTYPE "$ENV{KLLM_KDA_SSM_DTYPE}")
endif()

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(KDA_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping KDA kernel generation")
endif()

if(EXISTS ${KDA_MD5_FILE})
  file(READ ${KDA_MD5_FILE} PREV_MD5)
  calculate_files_md5("${KDA_ALL_FILES}" KDA_FILES_MD5)
  set(CURRENT_MD5 "${KDA_FILES_MD5}-${KDA_SSM_DTYPE}")
  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(KDA_NEED_REGENERATE FALSE)
    message(STATUS "KDA files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "KDA files MD5 changed, regenerating")
  endif()
endif()

if(KDA_NEED_REGENERATE)
  message(STATUS "Start generate KDA triton kernels")

  execute_process(COMMAND bash kda_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE KDA_RESULT_VARIABLE
    OUTPUT_VARIABLE KDA_RESULT_OUTPUT)

  if(KDA_RESULT_VARIABLE AND NOT KDA_RESULT_VARIABLE EQUAL 0)
    message(FATAL_ERROR "bash kda_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH} with error:\n${KDA_RESULT_OUTPUT}")
  endif()

  calculate_files_md5("${KDA_ALL_FILES}" NEW_MD5)
  file(WRITE ${KDA_MD5_FILE} "${NEW_MD5}-${KDA_SSM_DTYPE}")

  message(STATUS "Success generate KDA triton kernels")
endif()
endif()
