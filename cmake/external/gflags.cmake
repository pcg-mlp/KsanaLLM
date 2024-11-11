# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

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

    # Make the namespace of gflags be "google" instead of "gflags"
    set(GFLAGS_NAMESPACE "google")

    add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR})
endif()

message(STATUS "Google flags source directory: ${gflags_SOURCE_DIR}")
message(STATUS "Google flags binary directory: ${gflags_BINARY_DIR}")

include_directories(${gflags_BINARY_DIR}/include)
