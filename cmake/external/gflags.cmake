# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

if(TARGET gflags)
    return()
endif()

# When _GLIBCXX_USE_CXX11_ABI=0 (e.g. Kunlun/XPU torch), system gflags (compiled with new ABI)
# has incompatible mangled symbols.  Always build from source to inherit our project-wide ABI flag.
# This also avoids runtime "flag defined more than once" issues from mixed linkage.
set(_GFLAGS_FORCE_SOURCE OFF)
# Detect: if our compile defs include _GLIBCXX_USE_CXX11_ABI=0, system .so won't match.
get_directory_property(_cur_defs COMPILE_DEFINITIONS)
if("${_cur_defs}" MATCHES "_GLIBCXX_USE_CXX11_ABI=0")
  set(_GFLAGS_FORCE_SOURCE ON)
  message(STATUS "gflags: old ABI detected, forcing build from source for ABI compatibility")
endif()

# 优先使用系统已安装的 gflags，避免与 glog 等依赖的系统 gflags 产生
# 重复符号冲突（运行时 flag 重复注册导致 "flag was defined more than once"）。
# 仅在系统未安装 gflags 时，才从源码编译。
#
# NO_CMAKE_PACKAGE_REGISTRY: 忽略 CMake 用户包注册表
# (~/.cmake/packages/gflags/)。否则某次构建里 mooncake 子构建安装的私有
# gflags 会把指向其工作区 deps 目录的路径注册进去，导致后续无关构建在系统
# 本无 gflags 时却"找到了"一个陈旧/错配的 gflags，引发 include 路径与
# IMPORTED target 解析问题。只认系统路径与 CMAKE_PREFIX_PATH。
if(NOT _GFLAGS_FORCE_SOURCE)
  find_package(gflags QUIET NO_CMAKE_PACKAGE_REGISTRY)
endif()

# Source dir of a from-source gflags build, exported for downstream
# consumers (external/mooncake_tent.cmake installs it into a private
# prefix for the Mooncake tent sub-build's find_package(gflags)). Empty
# when system gflags is used — in that case the sub-build finds the same
# system gflags and needs no help. CACHE INTERNAL so the value survives
# regardless of which CMakeLists.txt later includes external/mooncake_tent.
set(KSANA_GFLAGS_SOURCE_DIR "" CACHE INTERNAL "gflags source dir for the Mooncake tent sub-build")

if(gflags_FOUND AND NOT _GFLAGS_FORCE_SOURCE)
    message(STATUS "Using system gflags: version ${GFLAGS_VERSION_STRING}, "
                   "include=${GFLAGS_INCLUDE_DIR}, libs=${GFLAGS_LIBRARIES}")
    # Create an INTERFACE target so that target_link_libraries(… gflags …)
    # works regardless of whether gflags was found as a system package or
    # built from source.
    if(NOT TARGET gflags)
        add_library(gflags INTERFACE)
        target_include_directories(gflags INTERFACE ${GFLAGS_INCLUDE_DIR})
        target_link_libraries(gflags INTERFACE ${GFLAGS_LIBRARIES})
    endif()
else()
    if(NOT gflags_FOUND)
      message(STATUS "System gflags not found, building from source")
    endif()
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
        # Build as static library to avoid conflicts with system gflags.so
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
        set(BUILD_gflags_LIB ON CACHE BOOL "" FORCE)

        add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR})

        # gflags_static inherits our compile_definitions (including _GLIBCXX_USE_CXX11_ABI=0)
        # Alias so downstream `target_link_libraries(... gflags)` resolves to the static target.
        if(TARGET gflags_static AND NOT TARGET gflags)
          add_library(gflags ALIAS gflags_static)
        endif()
    endif()

    message(STATUS "Google flags source directory: ${gflags_SOURCE_DIR}")
    message(STATUS "Google flags binary directory: ${gflags_BINARY_DIR}")

    include_directories(${gflags_BINARY_DIR}/include)
    set(KSANA_GFLAGS_SOURCE_DIR "${gflags_SOURCE_DIR}"
        CACHE INTERNAL "gflags source dir for the Mooncake tent sub-build")
endif()
