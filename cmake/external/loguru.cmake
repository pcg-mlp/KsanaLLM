# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(LOGURU_INSTALL_DIR ${THIRD_PARTY_PATH}/install/loguru)

FetchContent_Declare(
    loguru
    GIT_REPOSITORY https://github.com/emilk/loguru.git
    GIT_TAG v2.1.0
    GIT_SHALLOW TRUE
    SOURCE_DIR ${LOGURU_INSTALL_DIR}
)

add_definitions(-DLOGURU_USE_FMTLIB=1)
add_definitions(-DLOGURU_WITH_STREAMS=1)
FetchContent_GetProperties(loguru)
if(NOT loguru_POPULATED)
    FetchContent_Populate(loguru)

    add_library(loguru SHARED ${LOGURU_INSTALL_DIR}/loguru.cpp)

    target_link_libraries(loguru fmt)
endif()

include_directories(${LOGURU_INSTALL_DIR})
