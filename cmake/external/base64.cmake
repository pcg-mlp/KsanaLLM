# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(BASE64_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/base64)

FetchContent_Populate(download_base64
    GIT_REPOSITORY https://github.com/tobiaslocker/base64.git
    GIT_TAG master
    SOURCE_DIR ${BASE64_INCLUDE_DIR}
    SUBBUILD_DIR ${THIRD_PARTY_PATH}/tmp
    BINARY_DIR ${THIRD_PARTY_PATH}/tmp
)

include_directories(${BASE64_INCLUDE_DIR}/include)
