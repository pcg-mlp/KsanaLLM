# Copyright 2024 Tencent Inc.  All rights reserved.

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

option(WITH_CUDA "Enable Cuda" ON)
option(WITH_ACL "Enable Ascend" OFF)
option(WITH_TESTING "Enable testing" OFF)
option(CUDA_PTX_VERBOSE_INFO "build nvidia kernels with detailed ptx info" OFF)

if(WITH_CUDA)
  project(LLMKernels LANGUAGES CXX CUDA)
else()
  set(ASCEND_PRODUCT_TYPE "ascend910")
  set(ASCEND_CORE_TYPE "AiCore")
  set(ASCEND_RUN_MODE "ONBOARD")
  set(ASCEND_INSTALL_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/module)
  list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})

  # NOTE(karlluo): languages cce need configured cmake module before
  project(LLMKernels LANGUAGES CCE CXX)
endif()

find_package(Git QUIET)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD OUTPUT_VARIABLE COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/third_party)

# set compiler flags
set(CXX_STD "17" CACHE STRING "C++ standard")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -DWMMA")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DWMMA")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# set cmake output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(base)
include(nvidia)
include(ascend)

# set headers
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${CUDA_INC_DIRS}
  ${ACL_INC_DIRS}
)

# set linked libraries
link_directories(
  ${CUDA_LIB_DIRS}
  ${ACL_LIB_DIRS}
)

include(test)

# add subdirectory
add_subdirectory(csrc)
add_subdirectory(tests)
