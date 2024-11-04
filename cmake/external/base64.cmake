# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(BASE64_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/base64)

FetchContent_Populate(download_base64
    GIT_REPOSITORY https://github.com/tobiaslocker/base64.git
    # https://github.com/tobiaslocker/base64/commits/master/
    GIT_TAG 387b32f337b83d358ac1ffe574e596ba99c41d31
    SOURCE_DIR ${BASE64_INCLUDE_DIR}
)

include_directories(${BASE64_INCLUDE_DIR}/include)
