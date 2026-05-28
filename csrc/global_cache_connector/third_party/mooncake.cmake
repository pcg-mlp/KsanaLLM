# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
#
# Mooncake dependency for global_cache_connector (libmooncake_connector.so).
#
# 查找策略:
#   1. MOONCAKE_SOURCE_ROOT 已定义 → 使用用户指定路径，并校验 git tag
#   2. 否则 → FetchContent 下载 Mooncake 源码 + execute_process 编译
#
# 输出变量:
#   MOONCAKE_SOURCE_ROOT — Mooncake 源码根目录 (用于 include)
#   MOONCAKE_BUILD_ROOT  — Mooncake 编译目录 (用于 link)

set(MOONCAKE_REQUIRED_VERSION "v0.3.9")

if(DEFINED MOONCAKE_SOURCE_ROOT)
    message(STATUS "Using user-provided Mooncake: ${MOONCAKE_SOURCE_ROOT}")

    # 版本检查: 要求 MOONCAKE_SOURCE_ROOT 处于指定 tag
    execute_process(
        COMMAND git describe --tags --exact-match HEAD
        WORKING_DIRECTORY ${MOONCAKE_SOURCE_ROOT}
        OUTPUT_VARIABLE _mc_tag
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _mc_tag_rc
    )
    if(_mc_tag_rc EQUAL 0)
        if(NOT _mc_tag STREQUAL MOONCAKE_REQUIRED_VERSION)
            message(FATAL_ERROR
                "Mooncake version mismatch: found '${_mc_tag}', required '${MOONCAKE_REQUIRED_VERSION}'.\n"
                "Please: cd ${MOONCAKE_SOURCE_ROOT} && git checkout ${MOONCAKE_REQUIRED_VERSION}")
        endif()
        message(STATUS "Mooncake version verified: ${_mc_tag}")
    else()
        execute_process(
            COMMAND git describe --tags HEAD
            WORKING_DIRECTORY ${MOONCAKE_SOURCE_ROOT}
            OUTPUT_VARIABLE _mc_desc
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        message(WARNING
            "Cannot determine exact Mooncake tag (got '${_mc_desc}'). "
            "Required '${MOONCAKE_REQUIRED_VERSION}'. Proceeding, but version mismatch may cause RPC errors.")
    endif()
else()
    # 自动拉取 Mooncake 源码并编译
    message(STATUS "MOONCAKE_SOURCE_ROOT not set, fetching Mooncake ${MOONCAKE_REQUIRED_VERSION} from source")
    include(FetchContent)

    set(MOONCAKE_INSTALL_DIR ${THIRD_PARTY_PATH}/install/mooncake)

    FetchContent_Declare(
        mooncake_src
        GIT_REPOSITORY https://github.com/kvcache-ai/Mooncake.git
        GIT_TAG        ${MOONCAKE_REQUIRED_VERSION}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        GIT_SUBMODULES_RECURSE TRUE
        SOURCE_DIR     ${MOONCAKE_INSTALL_DIR}
    )

    FetchContent_GetProperties(mooncake_src)
    if(NOT mooncake_src_POPULATED)
        FetchContent_Populate(mooncake_src)
    endif()

    set(MOONCAKE_SOURCE_ROOT ${mooncake_src_SOURCE_DIR})
    set(_mc_build_dir "${MOONCAKE_SOURCE_ROOT}/build")

    # 检查是否已编译 (避免重复编译)
    if(NOT EXISTS "${_mc_build_dir}/mooncake-store/src/libmooncake_store.a")
        message(STATUS "Building Mooncake in ${_mc_build_dir} ...")
        file(MAKE_DIRECTORY ${_mc_build_dir})

        execute_process(
            COMMAND ${CMAKE_COMMAND} ..
                -DUSE_CUDA=ON
                -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
                -DBUILD_TESTING=OFF
                -DENABLE_TESTS=OFF
                -DMOONCAKE_BUILD_TESTS=OFF
                -DMOONCAKE_ENABLE_TESTS=OFF
                -DBUILD_UNIT_TESTS=OFF
                -DBUILD_TEST=OFF
            WORKING_DIRECTORY ${_mc_build_dir}
            RESULT_VARIABLE _mc_cmake_rc
        )
        if(NOT _mc_cmake_rc EQUAL 0)
            message(FATAL_ERROR "Mooncake cmake configure failed (exit code ${_mc_cmake_rc})")
        endif()

        include(ProcessorCount)
        ProcessorCount(NPROC)
        if(NPROC EQUAL 0)
            set(NPROC 4)
        endif()

        execute_process(
            COMMAND make -j${NPROC}
            WORKING_DIRECTORY ${_mc_build_dir}
            RESULT_VARIABLE _mc_make_rc
        )
        if(NOT _mc_make_rc EQUAL 0)
            message(FATAL_ERROR "Mooncake build failed (exit code ${_mc_make_rc})")
        endif()

        message(STATUS "Mooncake build completed successfully")
    else()
        message(STATUS "Mooncake already built in ${_mc_build_dir}, skipping rebuild")
    endif()
endif()

# 设置 MOONCAKE_BUILD_ROOT (如果用户未显式指定)
if(NOT DEFINED MOONCAKE_BUILD_ROOT)
    set(MOONCAKE_BUILD_ROOT "${MOONCAKE_SOURCE_ROOT}/build")
endif()

# 验证关键产物存在
if(NOT EXISTS "${MOONCAKE_BUILD_ROOT}/mooncake-store/src/libmooncake_store.a")
    message(FATAL_ERROR
        "Mooncake build artifacts not found at ${MOONCAKE_BUILD_ROOT}.\n"
        "Expected: ${MOONCAKE_BUILD_ROOT}/mooncake-store/src/libmooncake_store.a\n"
        "Please build Mooncake first or check MOONCAKE_BUILD_ROOT.")
endif()

message(STATUS "Mooncake source: ${MOONCAKE_SOURCE_ROOT}")
message(STATUS "Mooncake build:  ${MOONCAKE_BUILD_ROOT}")
