# Copyright 2024 Tencent Inc.  All rights reserved.

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# support NVIDIA GPU
option(WITH_CUDA "Enable CUDA" ON)

# support Huawei NPU
option(WITH_ACL "Enable Ascend" OFF)

# support Tencent ZiXiao
option(WITH_TOPS "Enable Tops" OFF)

# support Baidu Kunlun XPU (Stage 0 scaffolding)
# 设计依据: tmp/kunlun-xpu-support-design-cn.md §3.2 / Stage 0
# 当前阶段 LLM_kernels 子项目不引入任何 XPU kernel —— 仅为通过 platform check,
# 让 add_subdirectory(3rdparty/LLM_kernels) 不再 fatal_error。
# Stage 6 自研 .xpu kernel 时再考虑在 csrc/kernels 下加 kunlun 子目录。
option(WITH_KUNLUN "Enable Kunlun XPU" OFF)

option(WITH_TESTING "Enable testing" OFF)
option(CUDA_PTX_VERBOSE_INFO "build nvidia kernels with detailed ptx info" OFF)

if(WITH_CUDA)
  project(LLMKernels LANGUAGES CXX CUDA)
  set(KLLM_CUDA_THREADS 2 CACHE STRING "nvcc --threads N per .cu file (0=disable, 1=low-disk, 2=default, 4+=fast)")
  if(KLLM_CUDA_THREADS GREATER 0)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--threads=${KLLM_CUDA_THREADS}>)
  endif()
elseif(WITH_ACL)
  set(ASCEND_PRODUCT_TYPE "ascend910")
  set(ASCEND_CORE_TYPE "AiCore")
  set(ASCEND_RUN_MODE "ONBOARD")
  set(ASCEND_INSTALL_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/module)
  set(SOC_VERSION "ascend910b1" CACHE STRING "system on chip type")
  set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
  set(RUN_MODE "npu" CACHE STRING "run mode: npu/sim/cpu")

  message(STATUS "SOC_VERSION: ${SOC_VERSION}")
  message(STATUS "RUN_MODE: ${RUN_MODE}")
  message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
  message(STATUS "CCE_CMAKE_PATH: ${CCE_CMAKE_PATH}")
  message(STATUS "ASCEND_INSTALL_PATH: ${ASCEND_INSTALL_PATH}")
  message(STATUS "ASCEND_RUN_MODE: ${ASCEND_RUN_MODE}")
  message(STATUS "ASCEND_CORE_TYPE: ${ASCEND_CORE_TYPE}")
  message(STATUS "ASCEND_PRODUCT_TYPE: ${ASCEND_PRODUCT_TYPE}")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)

  if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
  else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the compiler package is installed.")
  endif()

  include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
  list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})

  # NOTE(karlluo): languages cce need configured cmake module before
  project(LLMKernels LANGUAGES CCE CXX)
elseif(WITH_TOPS)
  set(ARCH "gcu300")
  project(LLMKernels LANGUAGES CXX)
elseif(WITH_KUNLUN)
  # Stage 0: 仅声明 CXX 工程, 不引入任何昆仑 kernel 源.
  # XPU_ARCH 由顶层 KsanaLLM/CMakeLists.txt 设置 (kl3), 这里不重复 set.
  project(LLMKernels LANGUAGES CXX)
else()
  message(FATAL_ERROR "Support platform is not selected. Select NVIDIA GPU with -DWITH_CUDA=ON, Huawei NPU with -DWITH_ACL=ON, Tencent ZiXiao with -DWITH_TOPS=ON or Baidu Kunlun XPU with -DWITH_KUNLUN=ON")
endif()

find_package(Git QUIET)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD OUTPUT_VARIABLE COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)

link_directories(/usr/lib64)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_SOURCE_DIR}/cmake)
set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/third_party)

# set compiler flags
set(CXX_STD "17" CACHE STRING "C++ standard")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -DWMMA")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DWMMA -gdwarf-4 -gstrict-dwarf")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0 -gdwarf-4 -gstrict-dwarf")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -gdwarf-4 -gstrict-dwarf")

if(WITH_TESTING)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

# set cmake output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(base)
include(nvidia)
include(ascend)
# TODO(kunlun-stage0): flashinfer 仅 CUDA 后端使用 (其 cmake 文件 add_subdirectory 已用 if(WITH_CUDA) 守门),
# 但 flashinfer.cmake 顶部的 FetchContent_Populate 是无条件的, 在无网环境会永久卡死. 这里收紧一层:
# 非 CUDA 后端 (NPU/ZiXiao/Kunlun) 直接跳过整个 flashinfer.cmake.
# 这是 LLM_kernels 上游设计 bug 的临时绕过, 上游修复后可回退此处. 见 tmp/kunlun-stage0-todo.md.
if(WITH_CUDA)
  include(flashinfer)
endif()

if(WITH_CUDA)
  include(fmt)
  include(yaml-cpp)
endif()

# fetch 3rdparty
if(GIT_FOUND)
  message(STATUS "Running submodule update to fetch half")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/half
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/half failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
  endif()
endif()

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

set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_VERSION)

if(TORCH_VERSION VERSION_LESS "1.5.0")
  message(FATAL_ERROR "PyTorch >= 1.5.0 is needed for TorchScript mode.")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;print(os.path.dirname(torch.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_DIR)

if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Torch config Error.")
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})

# find_package(Torch) 会触发 Caffe2Config -> public/cuda.cmake -> enable_language(CUDA),
# TODO(kunlun-stage0): 在没有 nvcc / nvcc 与 GCC 不兼容 / 我们根本不要 CUDA 的环境会直接 fatal.
# Kunlun (XPU) 后端: LLM_kernels 子项目此阶段不引入任何 kernel, 不依赖 libtorch C++ 接口,
# 故跳过整段 find_package(Torch) 及其 cleanup. 后续若需 host 侧使用 torch C++ API,
# 应通过 xpytorch 包提供的等价 cmake 配置, 而非原生 Torch.
# 这是 LLM_kernels 上游设计 bug (REQUIRED + 强制 enable CUDA) 的临时绕过, 上游修复后可回退.
# 见 tmp/kunlun-stage0-todo.md.
if(WITH_CUDA OR WITH_ACL OR WITH_TOPS)
  # Save our CMAKE_CUDA_ARCHITECTURES before find_package(Torch) — Torch may overwrite it
  set(_SAVED_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  set(_SAVED_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  set(_SAVED_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
  set(_SAVED_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")

  find_package(Torch REQUIRED)

  # Restore CMAKE_CUDA_ARCHITECTURES to our own setting (Torch may have overwritten it)
  set(CMAKE_CUDA_ARCHITECTURES "${_SAVED_CUDA_ARCHITECTURES}")
  message(STATUS "Restored CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

  # Restore CMAKE_CUDA_FLAGS (remove any -gencode Torch injected)
  set(CMAKE_CUDA_FLAGS "${_SAVED_CUDA_FLAGS}")
  set(CMAKE_CUDA_FLAGS_RELEASE "${_SAVED_CUDA_FLAGS_RELEASE}")
  set(CMAKE_CUDA_FLAGS_DEBUG "${_SAVED_CUDA_FLAGS_DEBUG}")
  message(STATUS "Restored CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

  # Clean torch_cuda INTERFACE_COMPILE_OPTIONS to prevent -gencode propagation
  foreach(_torch_tgt torch_cuda torch_cuda_cu torch_cuda_cpp)
    if(TARGET ${_torch_tgt})
      get_target_property(_opts ${_torch_tgt} INTERFACE_COMPILE_OPTIONS)
      if(_opts)
        list(FILTER _opts EXCLUDE REGEX "-gencode")
        set_target_properties(${_torch_tgt} PROPERTIES INTERFACE_COMPILE_OPTIONS "${_opts}")
      endif()
    endif()
  endforeach()
  message(STATUS "Cleaned -gencode flags from Torch CUDA targets")
else()
  message(STATUS "Skipping find_package(Torch) for non-CUDA/ACL/TOPS backend (current: WITH_KUNLUN=${WITH_KUNLUN})")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig;print(sysconfig.get_python_inc());"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PY_INCLUDE_DIR)

if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Python config Error.")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE USE_CXX11_ABI)
include_directories(${PY_INCLUDE_DIR})
include_directories(${TORCH_DIR}/include/torch/csrc/api/include/)
include_directories(${TORCH_DIR}/include/)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)


# add subdirectory
add_subdirectory(csrc)
add_subdirectory(tests)

if(WITH_CUDA)
  add_subdirectory(tools/search_best_gemm_algo)
endif()
