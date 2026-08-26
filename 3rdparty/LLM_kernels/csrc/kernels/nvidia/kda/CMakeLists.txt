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

ksana_triton_kernel_md5_file(kda KDA_MD5_FILE)

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
  message(STATUS "Queue generate KDA triton kernels")
  calculate_files_md5("${KDA_ALL_FILES}" NEW_MD5)
  ksana_enqueue_triton_gen(
    NAME kda
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    MD5_FILE ${KDA_MD5_FILE}
    MD5_VALUE "${NEW_MD5}-${KDA_SSM_DTYPE}"
    COMMAND bash kda_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH})
endif()
endif()
