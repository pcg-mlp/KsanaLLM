# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(ABSEIL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/abseil-cpp)
set(ABSL_ENABLE_INSTALL ON CACHE BOOL "install abseil" FORCE)
set(CMAKE_CXX_FLAGS_SAVED "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

FetchContent_Declare(absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG        20240722.0
    GIT_SHALLOW    TRUE
    SOURCE_DIR     ${ABSEIL_INSTALL_DIR}
)

FetchContent_MakeAvailable(absl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SAVED}")

message(STATUS "Abseil source directory: ${absl_SOURCE_DIR}")
message(STATUS "Abseil binary directory: ${absl_BINARY_DIR}")

include_directories(SYSTEM ${absl_SOURCE_DIR})
set(absl_DIR ${absl_SOURCE_DIR})
