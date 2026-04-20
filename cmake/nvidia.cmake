# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

if(CUDA_PTX_VERBOSE_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-fno-plt -ldl -g3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD} -DCUDA_PTX_FP8_F2FP_ENABLED")
# Compress embedded GPU fatbinary to reduce .o / .so size for multi-SM builds
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin --compress-all")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler --generate-line-info -Wall -DCUDA_PTX_FP8_F2FP_ENABLED")

# set CUDA related
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
set(SM_SETS 80 86 89 90 90a)

# Parse SM into a list for multi-SM support (e.g. -DSM=80,86,89,90,90a)
if(NOT DEFINED SM OR SM STREQUAL "")
  # SM should already be defined from top-level CMakeLists.txt
  set(SM_LIST ${SM_SETS})
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

# check if custom define SM
# NOTE: We only set CMAKE_CUDA_ARCHITECTURES here (not CMAKE_CUDA_FLAGS -gencode).
# This allows per-target CUDA_ARCHITECTURES overrides to work correctly for
# SM-specific kernels.
if(NOT DEFINED SM)
  foreach(SM_NUM IN LISTS SM_SETS)
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message(STATUS "Assign GPU architecture (sm=${SM_NUM})")
  endforeach()
else()
  # Use SM_LIST for both single and multi-SM
  foreach(SM_NUM IN LISTS SM_LIST)
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message(STATUS "Assign GPU architecture (sm=${SM_NUM})")
  endforeach()
endif()

set(CUDA_INC_DIRS
  ${CUDA_PATH}/include
  ${CUTLASS_HEADER_DIR}
)

set(CUDA_LIB_DIRS
  ${CUDA_PATH}/lib64
)

add_definitions("-DENABLE_CUDA")

# enable FP8
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(STATUS "CUDA version: ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
endif()

# DeepGEMM support - only for SM90/SM90a
set(ENABLE_DEEPSEEK_DEEPGEMM_FLAG 0)
if(HAS_SM90 OR HAS_SM90A)
  set(ENABLE_DEEPSEEK_DEEPGEMM_FLAG 1)
endif()

if(ENABLE_DEEPSEEK_DEEPGEMM_FLAG)
  message(STATUS "DeepGEMM support enabled (SM90/90a detected).")
else()
  message(STATUS "DeepGEMM support disabled (requires SM90/90a).")
endif()

add_compile_definitions(ENABLE_DEEPSEEK_DEEPGEMM=${ENABLE_DEEPSEEK_DEEPGEMM_FLAG})
set(ENABLE_DEEPSEEK_DEEPGEMM_FLAG ${ENABLE_DEEPSEEK_DEEPGEMM_FLAG} CACHE INTERNAL "Toggle DeepSeek DeepGEMM kernels" FORCE)

find_package(Git QUIET)
if(GIT_FOUND AND ENABLE_DEEPSEEK_DEEPGEMM_FLAG)
    set(DEEPGEMM_SOURCE_DIR ${CMAKE_BINARY_DIR}/third_party/DeepGEMM)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/third_party)

    if(NOT EXISTS ${DEEPGEMM_SOURCE_DIR})
        message(STATUS "Cloning DeepGEMM repository...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --recursive https://github.com/deepseek-ai/DeepGEMM.git ${DEEPGEMM_SOURCE_DIR}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third_party
            RESULT_VARIABLE GIT_CLONE_RESULT
        )
        if(NOT GIT_CLONE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to clone DeepGEMM repository.")
        endif()
    else()
        message(STATUS "DeepGEMM repository already exists, skipping clone.")
    endif()

    # Check if DeepGEMM is already built
    set(DEEPGEMM_BUILT_MARKER ${DEEPGEMM_SOURCE_DIR}/build)
    if(EXISTS ${DEEPGEMM_BUILT_MARKER})
        message(STATUS "DeepGEMM already built, skipping build process.")
    else()
        message(STATUS "Updating DeepGEMM submodules...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${DEEPGEMM_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMODULE_RESULT
        )
        if(NOT GIT_SUBMODULE_RESULT EQUAL 0)
            message(WARNING "Failed to update DeepGEMM submodules.")
        endif()
        message(STATUS "Checking out DeepGEMM to commit 0f5f2662027f0db05d4e3f6a94e56e2d8fc45c51")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} checkout 0f5f2662027f0db05d4e3f6a94e56e2d8fc45c51
            WORKING_DIRECTORY ${DEEPGEMM_SOURCE_DIR}
            RESULT_VARIABLE GIT_CHECKOUT_RESULT
        )
        if(NOT GIT_CHECKOUT_RESULT EQUAL 0)
            message(WARNING "Failed to checkout DeepGEMM to 0f5f2662027f0db05d4e3f6a94e56e2d8fc45c51, using current branch.")
        endif()

        message(STATUS "Building and installing DeepGEMM...")
        # Set DG_USE_LOCAL_VERSION=0 to skip git clean check in setup.py
        set(ENV{DG_USE_LOCAL_VERSION} "0")
        message(STATUS "Creating scripts/__init__.py for DeepGEMM...")
        file(WRITE ${DEEPGEMM_SOURCE_DIR}/scripts/__init__.py "")
        execute_process(
            COMMAND bash develop.sh
            WORKING_DIRECTORY ${DEEPGEMM_SOURCE_DIR}
            RESULT_VARIABLE DEEPGEMM_INSTALL_RESULT
        )
        if(NOT DEEPGEMM_INSTALL_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to build and install DeepGEMM.")
        endif()
        message(STATUS "DeepGEMM built and installed successfully.")
    endif()

    set(DEEPGEMM_SOURCE_DIR ${DEEPGEMM_SOURCE_DIR} CACHE PATH "DeepGEMM source directory" FORCE)
    set(DEEPGEMM_LIBRARY_ROOT_PATH ${DEEPGEMM_SOURCE_DIR}/deep_gemm CACHE PATH "DeepGEMM library root path" FORCE)
    set(DEEPGEMM_INCLUDE_DIRS
        ${DEEPGEMM_SOURCE_DIR}/csrc
        ${DEEPGEMM_SOURCE_DIR}/deep_gemm/include
        ${DEEPGEMM_SOURCE_DIR}/third-party/cutlass/include
        ${DEEPGEMM_SOURCE_DIR}/third-party/cutlass/tools/util/include
        ${DEEPGEMM_SOURCE_DIR}/third-party/fmt/include
        CACHE INTERNAL "DeepGEMM include directories")
endif()
