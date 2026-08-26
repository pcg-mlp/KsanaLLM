# Copyright 2026 Tencent Inc.  All rights reserved.

# For Triton/TileLang kernels,
# generation can be omitted for lower SM architectures.
if(HAS_SM90 OR HAS_SM90A)
file(GLOB_RECURSE MHC_ALL_FILES
  mhc_compile.py
  mhc_kernel.py
  autotuner.py
)

ksana_triton_kernel_md5_file(mhc MHC_MD5_FILE)

# TileLang needs libnvrtc symbols visible globally, otherwise kernel generation
# aborts with "libnvrtc symbols not found globally". Prefer an explicit
# MHC_LD_PRELOAD from the environment, otherwise auto-detect libnvrtc from the
# CUDA toolkit (never the stubs/ dir, whose stub carries no real symbols).
if(DEFINED ENV{MHC_LD_PRELOAD})
  set(MHC_LD_PRELOAD $ENV{MHC_LD_PRELOAD})
else()
  find_library(MHC_NVRTC_LIBRARY
    NAMES nvrtc libnvrtc.so.12
    HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_PATH}
    PATH_SUFFIXES lib64 lib targets/x86_64-linux/lib)
  if(MHC_NVRTC_LIBRARY)
    set(MHC_LD_PRELOAD ${MHC_NVRTC_LIBRARY})
  endif()
endif()

if(NOT Python3_EXECUTABLE)
  set(MHC_PYTHON_EXECUTABLE python)
else()
  set(MHC_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

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
  message(STATUS "Queue generate mhc tilelang kernel")
  calculate_files_md5("${MHC_ALL_FILES}" NEW_MD5)
  if(MHC_LD_PRELOAD)
    message(STATUS "MHC compile with LD_PRELOAD=${MHC_LD_PRELOAD}")
    ksana_enqueue_triton_gen(
      NAME mhc
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      MD5_FILE ${MHC_MD5_FILE}
      MD5_VALUE ${NEW_MD5}
      ENV "LD_PRELOAD=${MHC_LD_PRELOAD}"
      COMMAND ${MHC_PYTHON_EXECUTABLE} mhc_compile.py ${TRITON_KERNEL_PATH})
  else()
    ksana_enqueue_triton_gen(
      NAME mhc
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      MD5_FILE ${MHC_MD5_FILE}
      MD5_VALUE ${NEW_MD5}
      COMMAND ${MHC_PYTHON_EXECUTABLE} mhc_compile.py ${TRITON_KERNEL_PATH})
  endif()
endif()
endif()

# CUDA kernel static library
file(GLOB_RECURSE MHC_CU_SRCS *.cu)
list(FILTER MHC_CU_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_mhc STATIC ${MHC_CU_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_mhc PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_mhc PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_mhc PUBLIC -lcudart llm_kernels_nvidia_utils)
