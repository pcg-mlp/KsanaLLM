# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(HTTPLIB_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/httplib)

FetchContent_Populate(download_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.17.1
    SOURCE_DIR ${HTTPLIB_INCLUDE_DIR}
)

include_directories(${HTTPLIB_INCLUDE_DIR})
