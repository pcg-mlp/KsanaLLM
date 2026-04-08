# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# 优先使用系统已安装的 gflags，避免与 glog 等依赖的系统 gflags 产生
# 重复符号冲突（运行时 flag 重复注册导致 "flag was defined more than once"）。
# 仅在系统未安装 gflags 时，才从源码编译。
find_package(gflags QUIET)

if(gflags_FOUND)
    message(STATUS "Using system gflags: version ${GFLAGS_VERSION_STRING}, "
                   "include=${GFLAGS_INCLUDE_DIR}, target=${GFLAGS_LIBRARIES}")
    include_directories(${GFLAGS_INCLUDE_DIR})
else()
    message(STATUS "System gflags not found, building from source")
    include(FetchContent)

    set(GFLAGS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gflags)

    if(NOT DEFINED GFLAGS_VER)
        set(GFLAGS_VER 2.2.2)
    endif()
    set(GFLAGS_GIT_URL https://github.com/gflags/gflags/archive/v${GFLAGS_VER}.tar.gz)

    FetchContent_Declare(
        gflags
        URL        ${GFLAGS_GIT_URL}
        SOURCE_DIR ${GFLAGS_INSTALL_DIR}
    )

    FetchContent_GetProperties(gflags)
    if(NOT gflags_POPULATED)
        FetchContent_Populate(gflags)

        set(GFLAGS_NAMESPACE "google")

        add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR})
    endif()

    message(STATUS "Google flags source directory: ${gflags_SOURCE_DIR}")
    message(STATUS "Google flags binary directory: ${gflags_BINARY_DIR}")

    include_directories(${gflags_BINARY_DIR}/include)
endif()
