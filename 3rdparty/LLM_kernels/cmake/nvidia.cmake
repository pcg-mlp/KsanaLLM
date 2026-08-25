# Copyright 2024 Tencent Inc.  All rights reserved.

if(NOT WITH_CUDA)
  return()
endif()

find_package(CUDA 11.2 REQUIRED)

# enable FP8
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(STATUS "CUDA version: ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
endif()

if(NOT DEFINED SM)
  execute_process(COMMAND python ${PROJECT_SOURCE_DIR}/tools/get_nvidia_gpu_properties.py OUTPUT_VARIABLE SM OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Parse SM into a list for multi-SM support (e.g. -DSM=80,86,89,90,90a)
if(NOT DEFINED SM OR SM STREQUAL "")
  set(SM_LIST "${SM_SETS}")
elseif("${SM}" MATCHES ",")
  string(REPLACE "," ";" SM_LIST "${SM}")
else()
  set(SM_LIST "${SM}")
endif()

# Create helper flags for each SM architecture
set(HAS_SM80 FALSE)
set(HAS_SM86 FALSE)
set(HAS_SM89 FALSE)
set(HAS_SM90 FALSE)
set(HAS_SM90A FALSE)
foreach(_sm IN LISTS SM_LIST)
  if(_sm STREQUAL "80")
    set(HAS_SM80 TRUE)
  elseif(_sm STREQUAL "86")
    set(HAS_SM86 TRUE)
  elseif(_sm STREQUAL "89")
    set(HAS_SM89 TRUE)
  elseif(_sm STREQUAL "90")
    set(HAS_SM90 TRUE)
  elseif(_sm STREQUAL "90a")
    set(HAS_SM90A TRUE)
  endif()
endforeach()
message(STATUS "SM_LIST: ${SM_LIST}")
message(STATUS "HAS_SM80=${HAS_SM80} HAS_SM86=${HAS_SM86} HAS_SM89=${HAS_SM89} HAS_SM90=${HAS_SM90} HAS_SM90A=${HAS_SM90A}")

execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/triton_kernel_files)

# fetch 3rdparty
if(GIT_FOUND)
  message(STATUS "Running submodule update to fetch cutlass c3x")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/c3x/cutlass
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/c3x/cutlass failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
  endif()
endif()

# fetch 3rdparty
if(GIT_FOUND)
  message(STATUS "Running submodule update to fetch cutlass c4x")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/c4x/cutlass
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/c4x/cutlass failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
  endif()

  message(STATUS "Applying c4x cutlass patch")
  # Refer to https://github.com/NVIDIA/cutlass/pull/2786
  execute_process(COMMAND ${GIT_EXECUTABLE} apply ${PROJECT_SOURCE_DIR}/3rdparty/c4x_cutlass.patch
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/3rdparty/c4x/cutlass)
endif()

# fetch 3rdparty
if(GIT_FOUND)
  message(STATUS "Running submodule update to fetch hpc-ops")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/hpc-ops
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/hpc-ops failed with ${GIT_SUBMOD_RESULT}, please checkout hpc-ops submodule")
  endif()
endif()

if(CUDA_PTX_VERBOSE_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-fno-plt -ldl -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall --generate-line-info -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD} -DCUDA_PTX_FP8_F2FP_ENABLED")
# Compress embedded GPU fatbinary to reduce .o / .so size for multi-SM builds
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin --compress-all")

set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")

message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
message(STATUS "CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")

# set CUDA related
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
set(SM_SETS 80 86 89 90 90a)

# Use SM_LIST (already parsed above) to set CUDA architectures
# NOTE: We only set CMAKE_CUDA_ARCHITECTURES here (not CMAKE_CUDA_FLAGS -gencode).
# This allows per-target CUDA_ARCHITECTURES overrides to work correctly for
# SM-specific kernels (e.g., flash_mla only builds for 90a, blockwise_gemm for 90/90a).
foreach(SM_NUM IN LISTS SM_LIST)
  list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
  message(STATUS "Assign GPU architecture (sm=${SM_NUM})")
endforeach()
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# setting cutlass v3.x
add_library(cutlass_c3x INTERFACE)
set(CUTLASS_C3X_SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/c3x/cutlass)
set(CUTLASS_C3X_HEADER_DIR ${CUTLASS_C3X_SOURCE_DIR}/include)
set(CUTLASS_C3X_TOOLS_HEADER_DIR ${CUTLASS_C3X_SOURCE_DIR}/tools/util/include)
target_include_directories(cutlass_c3x INTERFACE
    ${CUTLASS_C3X_HEADER_DIR}
    ${CUTLASS_C3X_TOOLS_HEADER_DIR}
)
subproject_version(${CUTLASS_C3X_SOURCE_DIR} CUTLASS_C3X_VERSION)
set(CUTLASS_C3X_VERSION_SUB_LIST ${CUTLASS_C3X_VERSION})
string(REPLACE "." ";" CUTLASS_C3X_VERSION_SUB_LIST "${CUTLASS_C3X_VERSION}")
message(STATUS "cutlass c3x version is: ${CUTLASS_C3X_VERSION}")
list(GET CUTLASS_C3X_VERSION_SUB_LIST 0 CUTLASS_C3X_MAJOR_VERSION)
list(GET CUTLASS_C3X_VERSION_SUB_LIST 1 CUTLASS_C3X_MINOR_VERSION)
list(GET CUTLASS_C3X_VERSION_SUB_LIST 2 CUTLASS_C3X_PATCH_VERSION)
target_compile_definitions(cutlass_c3x INTERFACE
    CUTLASS_MAJOR_VERSION=${CUTLASS_C3X_MAJOR_VERSION}
    CUTLASS_MINOR_VERSION=${CUTLASS_C3X_MINOR_VERSION}
    CUTLASS_PATCH_VERSION=${CUTLASS_C3X_PATCH_VERSION}
)

# setting cutlass v4.x
add_library(cutlass_c4x INTERFACE)
set(CUTLASS_C4X_SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/c4x/cutlass)
set(CUTLASS_C4X_HEADER_DIR ${CUTLASS_C4X_SOURCE_DIR}/include)
set(CUTLASS_C4X_TOOLS_HEADER_DIR ${CUTLASS_C4X_SOURCE_DIR}/tools/util/include)
target_include_directories(cutlass_c4x INTERFACE
    ${CUTLASS_C4X_HEADER_DIR}
    ${CUTLASS_C4X_TOOLS_HEADER_DIR}
)
subproject_version(${CUTLASS_C4X_SOURCE_DIR} CUTLASS_C4X_VERSION)
set(CUTLASS_C4X_VERSION_SUB_LIST ${CUTLASS_C4X_VERSION})
string(REPLACE "." ";" CUTLASS_C4X_VERSION_SUB_LIST "${CUTLASS_C4X_VERSION}")
message(STATUS "cutlass c4x version is: ${CUTLASS_C4X_VERSION}")
list(GET CUTLASS_C4X_VERSION_SUB_LIST 0 CUTLASS_C4X_MAJOR_VERSION)
list(GET CUTLASS_C4X_VERSION_SUB_LIST 1 CUTLASS_C4X_MINOR_VERSION)
list(GET CUTLASS_C4X_VERSION_SUB_LIST 2 CUTLASS_C4X_PATCH_VERSION)
target_compile_definitions(cutlass_c4x INTERFACE
    CUTLASS_MAJOR_VERSION=${CUTLASS_C4X_MAJOR_VERSION}
    CUTLASS_MINOR_VERSION=${CUTLASS_C4X_MINOR_VERSION}
    CUTLASS_PATCH_VERSION=${CUTLASS_C4X_PATCH_VERSION}
)

# enable flashmla on hopper arch (check SM_LIST for multi-SM support)
if(HAS_SM90A)
  set(ENABLE_FLASH_MLA "TRUE")
endif()
message(STATUS "ENABLE_FLASH_MLA: ${ENABLE_FLASH_MLA}")

# setting flashmla
if(GIT_FOUND AND (ENABLE_FLASH_MLA STREQUAL "TRUE"))
  message(STATUS "Running submodule update to fetch FlashMLA")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/FlashMLA
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/FlashMLA failed with ${GIT_SUBMOD_RESULT}, please checkout FlashMLA submodule")
  endif()

  set(DEEPSEEK_FLASH_MLA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/FlashMLA/csrc)

  message(STATUS "Applying FlashMLA patch")
  execute_process(COMMAND ${GIT_EXECUTABLE} apply --check ${PROJECT_SOURCE_DIR}/3rdparty/flashmla.patch
    WORKING_DIRECTORY ${DEEPSEEK_FLASH_MLA_SOURCE_DIR}
    RESULT_VARIABLE FLASHMLA_PATCH_CHECK_RESULT)
  if(FLASHMLA_PATCH_CHECK_RESULT EQUAL "0")
    execute_process(COMMAND ${GIT_EXECUTABLE} apply ${PROJECT_SOURCE_DIR}/3rdparty/flashmla.patch
      WORKING_DIRECTORY ${DEEPSEEK_FLASH_MLA_SOURCE_DIR}
      RESULT_VARIABLE FLASHMLA_PATCH_RESULT)
    if(NOT FLASHMLA_PATCH_RESULT EQUAL "0")
      message(FATAL_ERROR "git apply flashmla.patch failed with ${FLASHMLA_PATCH_RESULT}")
    endif()
  else()
    message(STATUS "FlashMLA patch already applied, skipping")
  endif()

  file(GLOB_RECURSE DEEPSEEK_FLASH_MLA_SOURCES ${DEEPSEEK_FLASH_MLA_SOURCE_DIR}/*.cu)
  # exclude files in sm100 and cutlass
  list(FILTER DEEPSEEK_FLASH_MLA_SOURCES EXCLUDE REGEX "${DEEPSEEK_FLASH_MLA_SOURCE_DIR}/sm100/.*\\.cu$")
  list(FILTER DEEPSEEK_FLASH_MLA_SOURCES EXCLUDE REGEX "${DEEPSEEK_FLASH_MLA_SOURCE_DIR}/cutlass/.*$")
  set(DEEPSEEK_FLASH_MLA_INCLUDES
    ${DEEPSEEK_FLASH_MLA_SOURCE_DIR}
    ${DEEPSEEK_FLASH_MLA_SOURCE_DIR}/api
    ${DEEPSEEK_FLASH_MLA_SOURCE_DIR}/kerutils/include
    ${DEEPSEEK_FLASH_MLA_SOURCE_DIR}/sm90
    ${DEEPSEEK_FLASH_MLA_SOURCE_DIR}/sm90/decode/sparse_fp8)
endif()

set(CUDA_INC_DIRS
  ${CUDA_PATH}/include
)

# CUDA 13.0+ relocated CCCL headers (thrust, cub, nv) under a cccl/ subdirectory.
# Detect and expose the path so that #include <thrust/...> / <cub/...> resolves.
# On CUDA <= 12.x the headers stay at ${CUDA_PATH}/include and this block is a no-op.
if(EXISTS "${CUDA_PATH}/include/cccl/thrust/version.h")
  list(APPEND CUDA_INC_DIRS "${CUDA_PATH}/include/cccl")
  set(THRUST_INCLUDE_DIR "${CUDA_PATH}/include/cccl" CACHE PATH "Thrust headers")
  message(STATUS "Detected CCCL headers under ${CUDA_PATH}/include/cccl (CUDA 13.0+)")
  # Make the path visible to subdirectories configured before the directory-level
  # include_directories() call in CMakeLists.txt (e.g. flashinfer).
  include_directories(BEFORE "${CUDA_PATH}/include/cccl")
endif()

set(CUDA_LIB_DIRS
  ${CUDA_PATH}/lib64
)

# enable FP8
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(STATUS "CUDA version: ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
endif()
