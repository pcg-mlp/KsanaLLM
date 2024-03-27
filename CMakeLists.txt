# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
option(WITH_CUDA "Enable CUDA" ON)
option(WITH_ACL "Enable Ascend" OFF)
option(WITH_TESTING "Enable testing" OFF)

if(WITH_CUDA)
  project(ksana_llm LANGUAGES CXX CUDA)
else()
  project(LLMKernels LANGUAGES CXX)
endif()

if(NOT WITH_CUDA AND NOT WITH_ACL)
  message(FATAL_ERROR "WITH_CUDA=OFF and WITH_ACL=OFF is not allow")
endif()

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/third_party)

find_package(Git QUIET)

if(WITH_CUDA)
  # dedicate for Nvidia GPU
  option(CUDA_PTX_VERBOSE_INFO "build nvidia kernels with detailed ptx info" OFF)
  find_package(CUDA 11.8 REQUIRED)
  find_package(NCCL REQUIRED)

  if(NOT DEFINED SM OR "${SM}" STREQUAL "")
    message(STATUS "finding sm with ${PROJECT_SOURCE_DIR}/tools/get_nvidia_gpu_properties.py")
    execute_process(COMMAND python ${PROJECT_SOURCE_DIR}/tools/get_nvidia_gpu_properties.py OUTPUT_VARIABLE SM OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Auto detect SM is ${SM}")
  endif()

  include(FlashAttention)
endif()

include(LLM_kernels)
include(base)

# set compiler flags
set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -DWMMA -gdwarf-4 -gstrict-dwarf")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0 -gdwarf-4 -gstrict-dwarf")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -gdwarf-4 -gstrict-dwarf")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g3 -DWMMA -gdwarf-4 -gstrict-dwarf")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O3 -Werror=return-type -Wall -Wno-strict-aliasing -Wno-pointer-arith -Wno-ignored-attributes -Wno-deprecated -finline-functions -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0 -gdwarf-4 -gstrict-dwarf")
set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -gdwarf-4 -gstrict-dwarf")

if(WITH_CUDA)
  include(nvidia)
endif()

if(WITH_ACL)
  include(ascend)
endif()

# set cmake output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set include headers
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
)

# set headers
include_directories(
  ${COMMON_HEADER_DIRS}
  ${CUDA_INC_DIRS}
  ${ACL_INC_DIRS}
)

# set linked libraries
link_directories(
  ${CUDA_LIB_DIRS}
  ${ACL_LIB_DIRS}
)

if(WITH_TESTING)
  enable_testing()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# include torch
set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_VERSION)

if(TORCH_VERSION VERSION_LESS "1.5.0")
  message(FATAL_ERROR "PyTorch >= 1.5.0 is needed for TorchScript mode.")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_DIR)

if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Torch config Error.")
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PY_INCLUDE_DIR)

if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Python config Error.")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE USE_CXX11_ABI)
include_directories(${PY_INCLUDE_DIR})
include_directories(${TORCH_DIR}/include/torch/csrc/api/include/)
include_directories(${TORCH_DIR}/include/)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_subdirectory(3rdparty)
add_subdirectory(examples)
add_subdirectory(src)

# for integration test
if(WITH_TESTING)
  add_subdirectory(tests)
endif()
