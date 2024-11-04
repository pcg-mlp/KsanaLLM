# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(LOGURU_INSTALL_DIR ${THIRD_PARTY_PATH}/install/loguru)

FetchContent_Declare(
    loguru
    GIT_REPOSITORY https://github.com/whitelok/loguru.git
    GIT_TAG f63653183f69c5b8987a4415773ca64d9f3fc2f4
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
