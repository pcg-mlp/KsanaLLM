# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# DeepEP (Deep Expert Parallel) 模块

if(WITH_CUDA)
  # 查找 CUDA Toolkit（需要 cudaStream_t 等类型定义）
  find_package(CUDAToolkit REQUIRED)
  
  # NVSHMEM 路径配置
  set(NVSHMEM_ROOT "/usr/local/nvshmem")
  set(NVSHMEM_INCLUDE_DIR "${NVSHMEM_ROOT}/include")
  set(NVSHMEM_LIBRARY_DIR "${NVSHMEM_ROOT}/lib")
  
  # 查找 NVSHMEM 库
  find_library(NVSHMEM_LIBRARY
    NAMES nvshmem_host nvshmem
    PATHS ${NVSHMEM_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
  
  # 检查 NVSHMEM 是否存在
  if(NOT EXISTS "${NVSHMEM_ROOT}")
    message(WARNING "NVSHMEM not found at: ${NVSHMEM_ROOT}")
    message(WARNING "Skipping deepep module compilation")
    return()
  endif()

  if(NOT NVSHMEM_LIBRARY)
    message(WARNING "NVSHMEM library not found in: ${NVSHMEM_LIBRARY_DIR}")
    message(WARNING "Skipping deepep module compilation")
    return()
  endif()

  message(STATUS "Found NVSHMEM at: ${NVSHMEM_ROOT}")
  message(STATUS "NVSHMEM include: ${NVSHMEM_INCLUDE_DIR}")
  message(STATUS "NVSHMEM library: ${NVSHMEM_LIBRARY}")

  # DeepEP 预编译库路径配置
  set(DEEPEP_BASE_PATH "/usr/local/deepep/")
  set(DEEPEP_LIB_PATH "${DEEPEP_BASE_PATH}/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so")
  # DeepEP头文件路径：包含csrc/和csrc/kernels/目录
  # 这样可以直接 #include "config.hpp" 和 #include "kernels/configs.cuh"
  set(DEEPEP_INCLUDE_DIR "${DEEPEP_BASE_PATH}/csrc/" "${DEEPEP_BASE_PATH}/csrc/kernels/")

  # 检查 DeepEP 库是否存在
  if(NOT EXISTS "${DEEPEP_LIB_PATH}")
    message(WARNING "DeepEP library not found at: ${DEEPEP_LIB_PATH}")
    message(WARNING "Skipping deepep module compilation")
    return()
  endif()

  # 检查 DeepEP 头文件目录是否存在
  if(NOT EXISTS "${DEEPEP_BASE_PATH}/csrc/")
    message(WARNING "DeepEP include directory not found at: ${DEEPEP_BASE_PATH}/csrc/")
    message(WARNING "Skipping deepep module compilation")
    return()
  endif()

  message(STATUS "Found DeepEP library: ${DEEPEP_LIB_PATH}")
  message(STATUS "DeepEP include directories: ${DEEPEP_INCLUDE_DIR}")

  # 创建 IMPORTED 库用于链接预编译的 DeepEP .so 文件
  add_library(DeepEP::deep_ep SHARED IMPORTED)
  set_target_properties(DeepEP::deep_ep PROPERTIES
    IMPORTED_LOCATION "${DEEPEP_LIB_PATH}"
  )

  # 不再编译 deep_ep.cpp，直接使用 DeepEP 的 .so 文件
  # 创建一个 INTERFACE 库来传递 DeepEP 的依赖
  add_library(deepep INTERFACE)

  target_link_libraries(deepep
    INTERFACE
      DeepEP::deep_ep
      ${NVSHMEM_LIBRARY}
  )

  target_include_directories(deepep
    INTERFACE
      ${CMAKE_CURRENT_SOURCE_DIR}      # deepep目录本身对外可见
      ${CUDAToolkit_INCLUDE_DIRS}
      ${NVSHMEM_INCLUDE_DIR}
      ${DEEPEP_INCLUDE_DIR}            # DeepEP头文件
      ${TORCH_INCLUDE_DIRS}
  )

  # 定义宏表示 DeepEP 可用
  target_compile_definitions(deepep
    INTERFACE
      KSANA_WITH_DEEPEP=1
  )
else()
  message(STATUS "Skipping deepep module (requires CUDA)")
endif()
