# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(HTTPLIB_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/httplib)

FetchContent_Populate(download_httplib
	GIT_REPOSITORY	https://github.com/yhirose/cpp-httplib.git
	GIT_TAG			master
    SOURCE_DIR ${HTTPLIB_INCLUDE_DIR}
    SUBBUILD_DIR ${THIRD_PARTY_PATH}/tmp
    BINARY_DIR ${THIRD_PARTY_PATH}/tmp
)

include_directories(${HTTPLIB_INCLUDE_DIR})
