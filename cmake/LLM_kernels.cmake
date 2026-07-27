# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
# 主工程接入 3rdparty/LLM_kernels；可选远程静态库缓存（tools/llm_kernels_cache.py）。
#
# 低入侵时序（不改 3rdparty/LLM_kernels 内部 fetch/patch）：
#   1) 若已 USE_LLM_KERNELS_PREBUILT=ON → 直接 IMPORTED，return
#   2) 否则照常 add_subdirectory（由其自身 init half/cutlass/FlashMLA + patch）
#   3) add_subdirectory 结束后再算 key / try-download（deps 工作区已定）
#   4) HIT → 跳过 NVCC：HEADER_FILE_ONLY + stub，再由 apply_prebuilt 覆盖 .a
#   5) UPLOAD=1 → 注册 try-upload（仅未 HIT 时）
#
# 环境变量见 tools/llm_kernels_cache.py / help-env。
# 强制源码重编：-DUSE_LLM_KERNELS_PREBUILT=OFF
# =============================================================================

option(USE_LLM_KERNELS_PREBUILT "Skip LLM_kernels compilation, use prebuilt static libs" OFF)
set(LLM_KERNELS_PREBUILT_DIR "" CACHE PATH "Prebuilt LLM_kernels bundle directory (contains manifest.json)")

# prepare cutlass
message(STATUS "Running submodule build to fetch LLM_kernels")

function(_llm_kernels_env_truthy _var _out)
  set(_v "$ENV{${_var}}")
  string(TOUPPER "${_v}" _v)
  if(_v STREQUAL "1" OR _v STREQUAL "ON" OR _v STREQUAL "TRUE" OR _v STREQUAL "YES")
    set(${_out} TRUE PARENT_SCOPE)
  else()
    set(${_out} FALSE PARENT_SCOPE)
  endif()
endfunction()

# LLM_kernels.cmake 可能早于 find_package(Python3)
function(_llm_kernels_find_python _out)
  set(_py "")
  if(Python3_EXECUTABLE)
    set(_py "${Python3_EXECUTABLE}")
  elseif(PYTHON_EXECUTABLE)
    set(_py "${PYTHON_EXECUTABLE}")
  elseif(DEFINED ENV{PYTHON_PATH} AND NOT "$ENV{PYTHON_PATH}" STREQUAL "")
    set(_py "$ENV{PYTHON_PATH}")
  endif()
  if(NOT _py)
    find_program(_py NAMES python3 python)
  endif()
  set(${_out} "${_py}" PARENT_SCOPE)
endfunction()

function(_llm_kernels_collect_targets out_var dir)
  set(_acc "")
  get_property(_tgts DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  list(APPEND _acc ${_tgts})
  get_property(_subs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(_s IN LISTS _subs)
    _llm_kernels_collect_targets(_child "${_s}")
    list(APPEND _acc ${_child})
  endforeach()
  set(${out_var} "${_acc}" PARENT_SCOPE)
endfunction()

# HIT 后替换：不改目标名；源文件不参与编译；用包内 .a 覆盖最终产物（跳过 NVCC）
function(_llm_kernels_replace_static_with_prebuilt _prebuilt_dir)
  if(NOT EXISTS "${_prebuilt_dir}/manifest.json")
    message(WARNING "LLM_kernels cache HIT but manifest missing: ${_prebuilt_dir}")
    return()
  endif()
  _llm_kernels_find_python(_py)
  if(NOT _py)
    message(WARNING "LLM_kernels cache HIT but python3 not found; cannot replace with prebuilt")
    return()
  endif()
  execute_process(
    COMMAND ${_py} -c
      "import json,sys; m=json.load(open(sys.argv[1])); print(';'.join(t['name'] for t in m.get('static_targets',[])))"
      "${_prebuilt_dir}/manifest.json"
    OUTPUT_VARIABLE _names
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc
  )
  if(NOT _rc EQUAL 0 OR _names STREQUAL "")
    message(WARNING "LLM_kernels cache: failed to read static_targets from manifest")
    return()
  endif()
  # 每个 STATIC 保留空 stub，避免 0 个 .o 时 ar 失败。
  # POST_BUILD 只能在目标创建目录挂接，故用统一 apply_prebuilt 覆盖 .a。
  set(_stub "${CMAKE_BINARY_DIR}/llm_kernels_cache_stub.cpp")
  if(NOT EXISTS "${_stub}")
    file(WRITE "${_stub}" "void llm_kernels_cache_stub() {}\n")
  endif()

  # 中间 STATIC/OBJECT（如 moe_gemm_src）不在 manifest，也要跳过 NVCC
  _llm_kernels_collect_targets(_all_tgts "${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels")
  list(APPEND _all_tgts ${_names})
  list(REMOVE_DUPLICATES _all_tgts)
  foreach(_t IN LISTS _all_tgts)
    if(NOT TARGET ${_t})
      continue()
    endif()
    get_target_property(_ttype ${_t} TYPE)
    if(NOT (_ttype STREQUAL "STATIC_LIBRARY" OR _ttype STREQUAL "OBJECT_LIBRARY"))
      continue()
    endif()
    get_target_property(_srcs ${_t} SOURCES)
    if(_srcs)
      # 源文件属性默认只对本目录可见；必须带 TARGET_DIRECTORY
      set_source_files_properties(${_srcs} TARGET_DIRECTORY ${_t}
        PROPERTIES HEADER_FILE_ONLY TRUE)
    endif()
    if(_ttype STREQUAL "STATIC_LIBRARY")
      target_sources(${_t} PRIVATE "${_stub}")
    endif()
  endforeach()

  set(_replace_deps "")
  set(_replace_cmds "")
  foreach(_t IN LISTS _names)
    if(NOT TARGET ${_t})
      continue()
    endif()
    get_target_property(_ttype ${_t} TYPE)
    if(NOT _ttype STREQUAL "STATIC_LIBRARY")
      continue()
    endif()
    set(_prebuilt_a "${_prebuilt_dir}/lib/lib${_t}.a")
    if(NOT EXISTS "${_prebuilt_a}")
      message(WARNING "LLM_kernels cache: missing prebuilt archive ${_prebuilt_a}")
      continue()
    endif()
    list(APPEND _replace_deps ${_t})
    list(APPEND _replace_cmds
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_prebuilt_a}" "$<TARGET_FILE:${_t}>"
    )
  endforeach()
  if(_replace_cmds)
    add_custom_target(llm_kernels_cache_apply_prebuilt ALL
      ${_replace_cmds}
      COMMENT "LLM_kernels cache: replace static libs with prebuilt .a (skip NVCC)"
      VERBATIM
    )
    add_dependencies(llm_kernels_cache_apply_prebuilt ${_replace_deps})
  endif()
  message(STATUS "LLM_kernels remote cache HIT -> replace static libs from ${_prebuilt_dir} (skip NVCC)")
endfunction()

# ---------------------------------------------------------------------------
# 手动指定 prebuilt：完全跳过 add_subdirectory
# ---------------------------------------------------------------------------
if(USE_LLM_KERNELS_PREBUILT)
  message(STATUS "USE_LLM_KERNELS_PREBUILT=ON, loading prebuilt LLM_kernels from ${LLM_KERNELS_PREBUILT_DIR}")
  if(NOT COMMAND cc_test)
    function(cc_test TARGET_NAME)
    endfunction()
  endif()
  include(LLM_kernels_prebuilt)
  include_directories(SYSTEM 3rdparty/LLM_kernels)
  return()
endif()

# ---------------------------------------------------------------------------
# 源码路径：gflags 必须在 add_subdirectory 前就绪，再照常编译 LLM_kernels
# ---------------------------------------------------------------------------
# LLM_kernels/tools/search_best_gemm_algo includes gflags/gflags.h and links
# against `gflags`. Make sure this is the CMake target from our external module
# before the LLM_kernels subdirectory is added; otherwise CMake treats `gflags`
# as a plain linker name and no include directory is propagated.
include(external/gflags)

# Suppress all warnings from LLM_kernels compilation
set(CMAKE_CXX_FLAGS_SAVED "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS_SAVED "${CMAKE_C_FLAGS}")
set(CMAKE_CUDA_FLAGS_SAVED "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

add_subdirectory(3rdparty/LLM_kernels)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SAVED}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_SAVED}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_SAVED}")

include_directories(SYSTEM 3rdparty/LLM_kernels)

# ---------------------------------------------------------------------------
# cmake 完成后：deps 已就绪 → 算 key / try-download → HIT 则替换产物
# ---------------------------------------------------------------------------
set(_LLM_KERNELS_CACHE_HIT FALSE)
_llm_kernels_env_truthy(LLM_KERNELS_REMOTE_CACHE _LLM_KERNELS_AUTO_DOWNLOAD)
if(_LLM_KERNELS_AUTO_DOWNLOAD)
  _llm_kernels_find_python(_LLM_KERNELS_CACHE_PY)
  if(_LLM_KERNELS_CACHE_PY)
    set(_LLM_KERNELS_CACHE_ENV
      "${CMAKE_COMMAND}" -E env
      "SM=${SM}"
      "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    )
    if(DEFINED ENV{LLM_KERNELS_CACHE_DIR})
      list(APPEND _LLM_KERNELS_CACHE_ENV "LLM_KERNELS_CACHE_DIR=$ENV{LLM_KERNELS_CACHE_DIR}")
    endif()
    message(STATUS "LLM_kernels remote cache: try-download after configure (LLM_KERNELS_REMOTE_CACHE=1)")
    execute_process(
      COMMAND ${_LLM_KERNELS_CACHE_ENV}
        ${_LLM_KERNELS_CACHE_PY}
        ${CMAKE_SOURCE_DIR}/tools/llm_kernels_cache.py try-download
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      RESULT_VARIABLE _LLM_KERNELS_DL_RC
      OUTPUT_VARIABLE _LLM_KERNELS_DL_OUT
      ERROR_VARIABLE _LLM_KERNELS_DL_ERR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_LLM_KERNELS_DL_RC EQUAL 0 AND _LLM_KERNELS_DL_OUT MATCHES "LLM_KERNELS_PREBUILT_DIR=([^\n\r]+)")
      set(_LLM_KERNELS_HIT_DIR "${CMAKE_MATCH_1}")
      if(EXISTS "${_LLM_KERNELS_HIT_DIR}/manifest.json")
        set(LLM_KERNELS_PREBUILT_DIR "${_LLM_KERNELS_HIT_DIR}" CACHE PATH
          "Prebuilt LLM_kernels bundle directory (contains manifest.json)" FORCE)
        _llm_kernels_replace_static_with_prebuilt("${_LLM_KERNELS_HIT_DIR}")
        set(_LLM_KERNELS_CACHE_HIT TRUE)
      endif()
    elseif(_LLM_KERNELS_DL_RC EQUAL 2)
      message(STATUS "LLM_kernels remote cache MISS -> will compile from source")
    elseif(NOT _LLM_KERNELS_DL_RC EQUAL 0)
      message(WARNING "LLM_kernels try-download failed (rc=${_LLM_KERNELS_DL_RC}), compile from source")
      if(_LLM_KERNELS_DL_ERR)
        message(STATUS "${_LLM_KERNELS_DL_ERR}")
      endif()
    endif()
  else()
    message(WARNING "LLM_KERNELS_REMOTE_CACHE=1 but python3 not found; skip auto download")
  endif()
endif()

# ---------------------------------------------------------------------------
# upload：仅未 HIT 替换时注册（编的是真源码）
# ---------------------------------------------------------------------------
set(LLM_KERNELS_REMOTE_CACHE_UPLOAD_ENABLED OFF CACHE INTERNAL "")
_llm_kernels_env_truthy(LLM_KERNELS_REMOTE_CACHE_UPLOAD _LLM_KERNELS_AUTO_UPLOAD)
if(_LLM_KERNELS_AUTO_UPLOAD AND NOT _LLM_KERNELS_CACHE_HIT)
  _llm_kernels_find_python(_LLM_KERNELS_CACHE_PY)
  if(NOT _LLM_KERNELS_CACHE_PY)
    message(WARNING "LLM_KERNELS_REMOTE_CACHE_UPLOAD=1 but python3 not found; skip auto upload")
  else()
    _llm_kernels_collect_targets(_LLM_KERNELS_ALL_TGTS "${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels")
    set(_LLM_KERNELS_UPLOAD_DEPS "")
    foreach(_t IN LISTS _LLM_KERNELS_ALL_TGTS)
      if(TARGET ${_t})
        get_target_property(_ttype ${_t} TYPE)
        if(_ttype STREQUAL "STATIC_LIBRARY")
          list(APPEND _LLM_KERNELS_UPLOAD_DEPS ${_t})
        endif()
      endif()
    endforeach()

    # 与 try-download 一致：可选透传 LLM_KERNELS_CACHE_DIR，避免 upload/download 写到不同目录
    set(_LLM_KERNELS_UPLOAD_ENV
      "SM=${SM}"
      "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    )
    if(DEFINED ENV{LLM_KERNELS_CACHE_DIR})
      list(APPEND _LLM_KERNELS_UPLOAD_ENV "LLM_KERNELS_CACHE_DIR=$ENV{LLM_KERNELS_CACHE_DIR}")
    endif()
    add_custom_target(llm_kernels_remote_cache_upload ALL
      COMMAND ${CMAKE_COMMAND} -E env
        ${_LLM_KERNELS_UPLOAD_ENV}
        ${_LLM_KERNELS_CACHE_PY}
        ${CMAKE_SOURCE_DIR}/tools/llm_kernels_cache.py try-upload
        --build-dir ${CMAKE_BINARY_DIR}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Upload LLM_kernels remote cache (LLM_KERNELS_REMOTE_CACHE_UPLOAD=1)"
      VERBATIM
    )
    if(_LLM_KERNELS_UPLOAD_DEPS)
      add_dependencies(llm_kernels_remote_cache_upload ${_LLM_KERNELS_UPLOAD_DEPS})
    endif()
    set(LLM_KERNELS_REMOTE_CACHE_UPLOAD_ENABLED ON CACHE INTERNAL "")
    message(STATUS "LLM_kernels remote cache: will try-upload after kernel libs build (${CMAKE_BINARY_DIR})")
  endif()
endif()
