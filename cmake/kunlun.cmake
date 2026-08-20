# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# XPU_ARCH (kl3/kl4) 必须由顶层 CMakeLists.txt 在 include(kunlun) 之前设置，
# 否则 add_definitions 展开为 -DXPU_ARCH_，静默失效。
if(NOT DEFINED XPU_ARCH OR "${XPU_ARCH}" STREQUAL "")
  message(FATAL_ERROR "XPU_ARCH must be set by parent CMakeLists.txt before include(kunlun). Expected values: kl3 / kl4.")
endif()

# --- 1. Python 解释器 + torch_xmlir 根目录 ---
# _PYBIN 从顶层 PYTHON_PATH 变量取，缺省 "python"。后续各节 find_spec 探测均复用它。
# XMLIR_DIR 优先从环境变量读（逃生口），否则用 importlib.util.find_spec 探测。
# find_spec 不触发 import hook，避免 stdout 被污染导致路径解析出错。
if(NOT DEFINED PYTHON_PATH OR "${PYTHON_PATH}" STREQUAL "")
  set(_PYBIN "python")
else()
  set(_PYBIN "${PYTHON_PATH}")
endif()

if(NOT "$ENV{XMLIR_DIR}" STREQUAL "")
  set(XMLIR_DIR "$ENV{XMLIR_DIR}")
  message(STATUS "Kunlun: using XMLIR_DIR from env = ${XMLIR_DIR}")
else()
  execute_process(
    COMMAND ${_PYBIN} -c "import importlib.util;spec=importlib.util.find_spec('torch_xmlir');print(spec.submodule_search_locations[0],end='')"
    OUTPUT_VARIABLE XMLIR_DIR
    RESULT_VARIABLE _xmlir_rc
    ERROR_QUIET
  )
  if(NOT _xmlir_rc EQUAL 0 OR "${XMLIR_DIR}" STREQUAL "")
    message(WARNING
      "Kunlun: cannot locate torch_xmlir python package via find_spec. "
      "Fix: pip install xmlir, or export XMLIR_DIR=/path/to/torch_xmlir.")
    set(XMLIR_DIR "/nonexistent/torch_xmlir")  # 让后续 EXISTS 检查 WARNING 一致
  else()
    message(STATUS "Kunlun: detected XMLIR_DIR = ${XMLIR_DIR}")
  endif()
endif()

# --- 2. XHPC 版本号 ---
# XHPC_VERSION_DATE 以宏形式注入，供 XHPC header 内部 #if 检查用。
# 优先从 SDK 的 _versions.txt 读取，否则用 CACHE 默认值。
set(XHPC_VERSION_DATE "20260305" CACHE STRING "XHPC date version, override if SDK upgraded")
if(EXISTS "${XMLIR_DIR}/_versions.txt")
  file(READ "${XMLIR_DIR}/_versions.txt" _versions_content)
  string(REGEX MATCH "XHPC:[ \t]*([0-9]+)" _ "${_versions_content}")
  if(CMAKE_MATCH_1)
    set(XHPC_VERSION_DATE "${CMAKE_MATCH_1}")
    message(STATUS "Kunlun: parsed XHPC_VERSION_DATE=${XHPC_VERSION_DATE} from _versions.txt")
  endif()
else()
  message(STATUS "Kunlun: _versions.txt not found, falling back to default XHPC_VERSION_DATE=${XHPC_VERSION_DATE}")
endif()

# --- 3. SDK include / lib 路径 ---
if(NOT EXISTS "${XMLIR_DIR}")
  message(WARNING "Kunlun: XMLIR_DIR=${XMLIR_DIR} does not exist. Linking/runtime will fail.")
endif()

# XPU_INC_DIRS: 五段 include，分别对应 runtime / XHPC 算法 / Flash Attention / BLAS / BKCL。
set(XPU_INC_DIRS
  ${XMLIR_DIR}/xre/include              # <xpu/runtime.h>: XPUStream / XPUEvent
  ${XMLIR_DIR}/xhpc/xdnn/include        # <xpu/refactor/...>: XHPC algorithms
  ${XMLIR_DIR}/xhpc/xfa/include         # <flash_api.h> / <flash_infer_api.h>: Flash Attention
  ${XMLIR_DIR}/xhpc/xblas/include       # <xblas_legacy_api.h>: bf16/fp16 GEMM (fc_fusion)
  ${XMLIR_DIR}/xccl/include             # <bkcl.h> / <xpu/bkcl.h>
  ${XMLIR_DIR}/include                  # XMLIR 自己的 header (兜底)
)

# XPU_LIB_DIRS: SDK 根目录存放大多数 .so；xre/so 存放 xpurt/cudart。
set(XPU_LIB_DIRS
  ${XMLIR_DIR}                          # libxpuapi / libbkcl / libxpu_dnn ...
  ${XMLIR_DIR}/xre/so                   # libxpurt / libcudart (XPU runtime)
)

# --- 3.1 infer_ops (xtorch_ops) ---
# 头文件由镜像提供，默认位于 /opt/klx_deps/infer_ops/include/。
# libapiinfer.so 由 xtorch_ops pip 包提供，通过 find_spec 自动定位。
# 逃生口：export INFER_OPS_INC_DIR=/path/to/infer_ops/include
#         export INFER_OPS_LIB_DIR=/path/to/dir/containing/libapiinfer.so

if(NOT "$ENV{INFER_OPS_INC_DIR}" STREQUAL "")
  set(_INFER_OPS_INC "$ENV{INFER_OPS_INC_DIR}")
  message(STATUS "Kunlun: infer_ops include dir from env = ${_INFER_OPS_INC}")
else()
  set(_INFER_OPS_INC "/opt/klx_deps/infer_ops/include")
endif()

if(NOT "$ENV{INFER_OPS_LIB_DIR}" STREQUAL "")
  set(_INFER_OPS_LIB_DIR "$ENV{INFER_OPS_LIB_DIR}")
  message(STATUS "Kunlun: infer_ops lib dir from env = ${_INFER_OPS_LIB_DIR}")
else()
  execute_process(
    COMMAND ${_PYBIN} -c
      "import importlib.util; spec=importlib.util.find_spec('xtorch_ops'); print(spec.submodule_search_locations[0],end='')"
    OUTPUT_VARIABLE _xtorch_ops_dir
    RESULT_VARIABLE _xtorch_ops_rc
    ERROR_QUIET
  )
  if(_xtorch_ops_rc EQUAL 0 AND NOT "${_xtorch_ops_dir}" STREQUAL "")
    set(_INFER_OPS_LIB_DIR "${_xtorch_ops_dir}")
    message(STATUS "Kunlun: detected xtorch_ops at ${_xtorch_ops_dir}")
  endif()
endif()

if(EXISTS "${_INFER_OPS_INC}/infer_ops.h"
   AND DEFINED _INFER_OPS_LIB_DIR
   AND EXISTS "${_INFER_OPS_LIB_DIR}/libapiinfer.so")
  list(APPEND XPU_INC_DIRS ${_INFER_OPS_INC})
  list(APPEND XPU_LIB_DIRS ${_INFER_OPS_LIB_DIR})
  message(STATUS "Kunlun: enabled infer_ops (headers=${_INFER_OPS_INC}, so=${_INFER_OPS_LIB_DIR})")
else()
  message(FATAL_ERROR
    "Kunlun: libapiinfer.so not found. Run: pip install xtorch_ops\n"
    "  or set INFER_OPS_LIB_DIR=/path/to/dir/containing/libapiinfer.so")
endif()

# --- 4. 链接库列表 (csrc/device/CMakeLists.txt 使用) ---
# 链接顺序：runtime (xpurt/cudart) 在末尾，确保符号解析正确。
set(XPU_SHARED_LIBS
  xpuapi
  bkcl
  xpu_dnn
  xpu_blas
  xpu_flash_attention
  xpurt
  xpuml
  cudart
  apiinfer
  xdnn_pytorch
  XMLIRRuntime
)

# --- 5. 编译宏 + 编译选项 ---
# ENABLE_KUNLUN: KsanaLLM 内部 device 类型分支依赖此宏
# XPU_ARCH_${XPU_ARCH}: 算子选择（如区分 kl3/kl4）
# XHPC_VERSION_DATE: xinfer 实测必须传入，用于 XHPC header 内部版本检查
add_definitions(
  "-DENABLE_KUNLUN"
  "-DXPU_ARCH_${XPU_ARCH}"
  "-DXHPC_VERSION_DATE=${XHPC_VERSION_DATE}"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "Kunlun: XPU_INC_DIRS    = ${XPU_INC_DIRS}")
message(STATUS "Kunlun: XPU_LIB_DIRS    = ${XPU_LIB_DIRS}")
message(STATUS "Kunlun: XPU_SHARED_LIBS = ${XPU_SHARED_LIBS}")
message(STATUS "Kunlun: XPU_ARCH        = ${XPU_ARCH}")

# --- 6. Torch include 路径 ---
# Kunlun 不走 find_package(Torch)，用 find_spec 手动定位，避免触发 import hook。
execute_process(
  COMMAND ${_PYBIN} -c "import importlib.util;spec=importlib.util.find_spec('torch');print(spec.submodule_search_locations[0],end='')"
  OUTPUT_VARIABLE _TORCH_DIR
  RESULT_VARIABLE _torch_rc
  ERROR_QUIET
)
if(_torch_rc EQUAL 0)
  string(STRIP "${_TORCH_DIR}" _TORCH_DIR)
  set(_TORCH_INC "${_TORCH_DIR}/include")
endif()

if(_torch_rc EQUAL 0 AND EXISTS "${_TORCH_INC}")
  set(TORCH_INCLUDE_DIRS
    ${_TORCH_INC}
    ${_TORCH_INC}/torch/csrc/api/include
  )
  message(STATUS "Kunlun: TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}")
else()
  message(WARNING "Kunlun: cannot detect torch include path, device_types.h may fail to compile.")
  set(TORCH_INCLUDE_DIRS "")
endif()

# --- 7. Torch 链接库（复用上面探测的 _TORCH_DIR）---
if(_torch_rc EQUAL 0 AND _TORCH_DIR)
  set(_TORCH_LIB_DIR "${_TORCH_DIR}/lib")
endif()
if(DEFINED _TORCH_LIB_DIR AND EXISTS "${_TORCH_LIB_DIR}")
  set(TORCH_LIBRARIES
    "${_TORCH_LIB_DIR}/libtorch.so"
    "${_TORCH_LIB_DIR}/libtorch_cpu.so"
    "${_TORCH_LIB_DIR}/libtorch_python.so"
    "${_TORCH_LIB_DIR}/libc10.so"
  )
  link_directories(${_TORCH_LIB_DIR})
  message(STATUS "Kunlun: TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
else()
  message(WARNING "Kunlun: cannot detect torch lib directory; TORCH_LIBRARIES will be empty.")
endif()

# --- 8. Python3 headers + libs ---
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# KUNLUN_PYTHON_LIBS 只链到 executable，不链 .so target。
set(KUNLUN_PYTHON_LIBS ${Python3_LIBRARIES})
