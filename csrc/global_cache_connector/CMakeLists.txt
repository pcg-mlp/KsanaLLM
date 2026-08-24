# Copyright 2026 Tencent Inc. All rights reserved.
#
# ==============================================================================

# ---------- 源文件收集 ----------
file(GLOB GLOBAL_CACHE_CONNECTOR_SRCS *.cpp)
list(FILTER GLOBAL_CACHE_CONNECTOR_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER GLOBAL_CACHE_CONNECTOR_SRCS EXCLUDE REGEX ".*test_connector.cpp")
list(FILTER GLOBAL_CACHE_CONNECTOR_SRCS EXCLUDE REGEX ".*bench_connector.cpp")
list(FILTER GLOBAL_CACHE_CONNECTOR_SRCS EXCLUDE REGEX ".*main.cpp")
# mooncake_connector.cpp 不再编入基础库，改为独立插件 .so
list(FILTER GLOBAL_CACHE_CONNECTOR_SRCS EXCLUDE REGEX ".*mooncake_connector.cpp")

file(GLOB GLOBAL_CACHE_CONNECTOR_HDRS *.h)
list(FILTER GLOBAL_CACHE_CONNECTOR_HDRS EXCLUDE REGEX ".*test.h")

# 引入第三方 mooncake-mock-store（仅提供 mooncake_store 动态库）
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/mooncake-mock-store
                 ${CMAKE_BINARY_DIR}/3rdparty/mooncake-mock-store)

# ========== 基础库: libglobal_cache_connector.so ==========
# 包含 factory + mock connector，不依赖 Mooncake。
# MooncakeConnector 通过 dlopen 按需加载。
add_library(global_cache_connector SHARED
    ${GLOBAL_CACHE_CONNECTOR_SRCS}
    ${PROJECT_SOURCE_DIR}/csrc/utils/logger.cpp
    ${PROJECT_SOURCE_DIR}/csrc/utils/status.cpp
)

set_target_properties(global_cache_connector PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    SOVERSION 1
    BUILD_RPATH "\$ORIGIN;\$ORIGIN/../lib"
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../lib"
    INSTALL_RPATH_USE_LINK_PATH OFF
)

target_include_directories(global_cache_connector
    PUBLIC
        ${PROJECT_SOURCE_DIR}/csrc
        ${PROJECT_SOURCE_DIR}/3rdparty/mooncake-mock-store/include
)

target_link_libraries(global_cache_connector
    PRIVATE loguru mooncake_store device dl
)

if(WITH_CUDA)
    target_link_libraries(global_cache_connector PRIVATE -lcudart)
endif()

# ========== 插件库: libmooncake_connector.so ==========
# 仅在 USE_MOONCAKE_REAL_STORE=ON 时编译，包含真实 Mooncake 后端实现。
# 运行时由 factory 通过 dlopen 加载。
# MOONCAKE_SOURCE_ROOT / MOONCAKE_BUILD_ROOT 由 third_party/mooncake.cmake 设置。

if(USE_MOONCAKE_REAL_STORE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/third_party/mooncake.cmake)
    message(STATUS "Mooncake real store backend enabled — building libmooncake_connector.so")

    add_library(mooncake_connector SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_connector.cpp
        ${PROJECT_SOURCE_DIR}/csrc/utils/logger.cpp
        ${PROJECT_SOURCE_DIR}/csrc/utils/status.cpp
    )

    # 仅 $ORIGIN：whl 把 libasio/libloguru 打进同目录，运行时无需 LD_LIBRARY_PATH
    set_target_properties(mooncake_connector PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        BUILD_RPATH "\$ORIGIN;\$ORIGIN/../lib"
        INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../lib"
        INSTALL_RPATH_USE_LINK_PATH OFF
        BUILD_WITH_INSTALL_RPATH TRUE
    )

    target_compile_definitions(mooncake_connector PRIVATE
        ENABLE_MOONCAKE_REAL_STORE GLOG_USE_GLOG_EXPORT YLT_ENABLE_IBV
        CONFIG_ERDMA USE_HTTP WITH_METRICS
    )
    target_compile_options(mooncake_connector PRIVATE -fcoroutines)

    target_include_directories(mooncake_connector PRIVATE
        ${PROJECT_SOURCE_DIR}/csrc
        ${PROJECT_SOURCE_DIR}/3rdparty/mooncake-mock-store/include
        ${MOONCAKE_SOURCE_ROOT}/mooncake-store/include
        ${MOONCAKE_SOURCE_ROOT}/mooncake-store/include/cachelib_memory_allocator/include
        ${MOONCAKE_SOURCE_ROOT}/mooncake-store/include/cachelib_memory_allocator/fake_include
        ${MOONCAKE_SOURCE_ROOT}/mooncake-store/include/cachelib_memory_allocator
        ${MOONCAKE_SOURCE_ROOT}/mooncake-transfer-engine/include
        ${MOONCAKE_SOURCE_ROOT}/mooncake-common/include
    )

    target_link_libraries(mooncake_connector PRIVATE
        loguru
        device
        ${MOONCAKE_BUILD_ROOT}/mooncake-store/src/libmooncake_store.a
        ${MOONCAKE_BUILD_ROOT}/mooncake-store/src/cachelib_memory_allocator/libcachelib_memory_allocator.a
        ${MOONCAKE_BUILD_ROOT}/mooncake-transfer-engine/src/libtransfer_engine.a
        ${MOONCAKE_BUILD_ROOT}/mooncake-common/src/libmooncake_common.a
        ${MOONCAKE_BUILD_ROOT}/mooncake-transfer-engine/src/common/base/libbase.a
        ${MOONCAKE_BUILD_ROOT}/mooncake-asio/libasio.so
        ibverbs glog numa pthread curl jsoncpp
    )

    if(WITH_CUDA)
        target_link_libraries(mooncake_connector PRIVATE -lcudart)
    endif()
else()
    message(STATUS "Mooncake real store backend disabled (default)")
endif()

# libcupti.so.11.8 is a versioned NEEDED entry in libtorch_cpu.so pulled in by
# KUNLUN_PYTHON_LIBS; resolve it at link time on Kunlun hosts without CUDA.
set(CUPTI_LIB_DIR "" CACHE PATH "Directory containing libcupti.so.11.8 (nvidia cuda_cupti conda package)")
if(WITH_KUNLUN AND NOT DEFINED _CUPTI_LIB_DIR)
  # libtorch_cpu.so requires libcupti.so.11.8 (versioned SONAME). The file at
  # /usr/local/cuda*/extras/CUPTI has SONAME 11.7 (mismatch). The nvidia conda
  # package has the correct SONAME. No .so symlink exists; search by filename.
  find_file(_CUPTI_LIB_FILE libcupti.so.11.8
    HINTS $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
          ${CUPTI_LIB_DIR}
    NO_DEFAULT_PATH)
  if(_CUPTI_LIB_FILE)
    get_filename_component(_CUPTI_LIB_DIR "${_CUPTI_LIB_FILE}" DIRECTORY)
  else()
    message(WARNING "libcupti.so.11.8 not found; Kunlun global_cache_bench link may fail. "
                    "Set -DCUPTI_LIB_DIR=<path> or install nvidia-cuda-cupti-cu11 in conda.")
  endif()
endif()

# ---------- Bench 可执行文件 ----------
add_executable(global_cache_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/global_cache_bench_test.cpp
)

target_link_libraries(global_cache_bench PRIVATE
    global_cache_connector
    yaml-cpp
    -lcudart
)
if(WITH_KUNLUN)
    # Kunlun Python 后端符号（xtorch_ops）在 embed 情况下由 executable 链
    target_link_libraries(global_cache_bench PRIVATE ${KUNLUN_PYTHON_LIBS})
    target_link_directories(global_cache_bench PRIVATE ${_CUPTI_LIB_DIR})
endif()

target_include_directories(global_cache_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/csrc
)

if(USE_MOONCAKE_REAL_STORE)
    target_compile_definitions(global_cache_bench PRIVATE
        ENABLE_MOONCAKE_REAL_STORE
    )
endif()

# ---------- 测试 ----------
if(WITH_TESTING)
    file(GLOB GCC_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
    list(FILTER GCC_TEST_SRCS EXCLUDE REGEX ".*bench_test\\.cpp")
    add_executable(global_cache_connector_test ${GCC_TEST_SRCS})
    target_compile_options(global_cache_connector_test PRIVATE -fno-access-control -g -O0)
    target_link_libraries(global_cache_connector_test
        global_cache_connector loguru gtest_main gtest -pthread)
    if(WITH_CUDA)
        target_link_libraries(global_cache_connector_test -lcudart)
    elseif(WITH_KUNLUN)
        target_link_libraries(global_cache_connector_test
            -L${XMLIR_DIR}/xre/so -lcudart
            ${KUNLUN_PYTHON_LIBS})
        target_link_directories(global_cache_connector_test PRIVATE ${_CUPTI_LIB_DIR})
    endif()
    add_test(NAME global_cache_connector_test
        COMMAND global_cache_connector_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
