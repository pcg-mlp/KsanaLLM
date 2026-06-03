# Copyright 2026 Tencent Inc.  All rights reserved.

# For Triton/TileLang kernels,
# generation can be omitted for lower SM architectures.
if(HAS_SM90 OR HAS_SM90A)
file(GLOB_RECURSE MHC_ALL_FILES
  mhc_compile.py
  mhc_kernel.py
  autotuner.py
)

set(MHC_MD5_SUM_DIR ${CMAKE_CURRENT_BINARY_DIR}/mhc_md5_sum)
set(MHC_MD5_FILE ${MHC_MD5_SUM_DIR}/.mhc.md5sums)

set(MHC_NEED_REGENERATE TRUE)

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(MHC_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping mhc kernel generation")
endif()

if(EXISTS ${MHC_MD5_FILE})
  file(READ ${MHC_MD5_FILE} PREV_MD5)

  calculate_files_md5("${MHC_ALL_FILES}" CURRENT_MD5)

  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(MHC_NEED_REGENERATE FALSE)
    message(STATUS "Files MD5 unchanged, skipping mhc regeneration")
  else()
    message(STATUS "Files MD5 changed, regenerating mhc")
    message(STATUS "Previous MD5: ${PREV_MD5}")
    message(STATUS "Current MD5: ${CURRENT_MD5}")
  endif()
endif()

if(MHC_NEED_REGENERATE)
  message(STATUS "Start generate mhc tilelang kernel")

  execute_process(COMMAND python mhc_compile.py ${TRITON_KERNEL_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE MHC_RESULT_VARIABLE OUTPUT_VARIABLE MHC_RESULT_OUTPUT)

  if(MHC_RESULT_VARIABLE AND NOT MHC_RESULT_VARIABLE EQUAL 0)
    message(FATAL_ERROR "python mhc_compile.py ${TRITON_KERNEL_PATH} with error:\n${MHC_RESULT_OUTPUT}")
  endif()

  calculate_files_md5("${MHC_ALL_FILES}" NEW_MD5)
  file(WRITE ${MHC_MD5_FILE} ${NEW_MD5})

  message(STATUS "Success generate mhc tilelang kernel")
endif()
endif()

# CUDA kernel static library
file(GLOB_RECURSE MHC_CU_SRCS *.cu)
list(FILTER MHC_CU_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_mhc STATIC ${MHC_CU_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_mhc PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_mhc PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_mhc PUBLIC -lcudart llm_kernels_nvidia_utils)
