# Copyright 2026 Tencent Inc.  All rights reserved.
#
# USE_LLM_KERNELS_PREBUILT=ON 时由 LLM_kernels.cmake include。
# LLM_KERNELS_PREBUILT_DIR 需含 manifest.json 与 lib/*.a。
# generate-cmake 生成 IMPORTED 同名目标；links 多项须用 ';'。
# IMPORTED 不带原 PUBLIC include，下方显式 SYSTEM include。

if(NOT LLM_KERNELS_PREBUILT_DIR)
  message(FATAL_ERROR "USE_LLM_KERNELS_PREBUILT=ON requires -DLLM_KERNELS_PREBUILT_DIR=...")
endif()

set(_LLM_KERNELS_MANIFEST "${LLM_KERNELS_PREBUILT_DIR}/manifest.json")
if(NOT EXISTS "${_LLM_KERNELS_MANIFEST}")
  message(FATAL_ERROR "LLM_kernels prebuilt manifest not found: ${_LLM_KERNELS_MANIFEST}")
endif()

message(STATUS "Using prebuilt LLM_kernels from ${LLM_KERNELS_PREBUILT_DIR}")

set(_LLM_KERNELS_PYTHON "${Python3_EXECUTABLE}")
if(NOT _LLM_KERNELS_PYTHON)
  find_program(_LLM_KERNELS_PYTHON NAMES python3 python PATHS /usr/local/bin /usr/bin ENV PATH)
endif()
if(NOT _LLM_KERNELS_PYTHON)
  message(FATAL_ERROR "USE_LLM_KERNELS_PREBUILT=ON requires python3 (Python3_EXECUTABLE empty)")
endif()
message(STATUS "LLM_kernels prebuilt generate-cmake python=${_LLM_KERNELS_PYTHON}")

set(_LLM_KERNELS_PREBUILT_GEN "${CMAKE_BINARY_DIR}/llm_kernels_prebuilt_targets.cmake")
execute_process(
  COMMAND ${_LLM_KERNELS_PYTHON}
    ${CMAKE_SOURCE_DIR}/tools/llm_kernels_cache.py generate-cmake
    --manifest ${_LLM_KERNELS_MANIFEST}
    --prebuilt-dir ${LLM_KERNELS_PREBUILT_DIR}
    --output ${_LLM_KERNELS_PREBUILT_GEN}
  RESULT_VARIABLE _LLM_KERNELS_PREBUILT_GEN_RESULT
  OUTPUT_VARIABLE _LLM_KERNELS_PREBUILT_GEN_OUTPUT
  ERROR_VARIABLE _LLM_KERNELS_PREBUILT_GEN_ERROR
)

if(NOT _LLM_KERNELS_PREBUILT_GEN_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Failed to generate prebuilt targets (rc=${_LLM_KERNELS_PREBUILT_GEN_RESULT}, python=${_LLM_KERNELS_PYTHON}):\n"
    "stdout: ${_LLM_KERNELS_PREBUILT_GEN_OUTPUT}\n"
    "stderr: ${_LLM_KERNELS_PREBUILT_GEN_ERROR}")
endif()

include(${_LLM_KERNELS_PREBUILT_GEN})
link_directories(${LLM_KERNELS_PREBUILT_DIR}/lib)

# IMPORTED 不带原 PUBLIC include，编译主工程 layers/kernels wrapper 仍要这些路径
include_directories(SYSTEM
  ${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels
  ${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels/csrc/kernels/nvidia/others/sglang/main/include
  ${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels/3rdparty/c3x/cutlass/include
  ${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels/3rdparty/c4x/cutlass/include
)
message(STATUS "Loaded prebuilt LLM_kernels targets from manifest")
