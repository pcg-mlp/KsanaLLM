# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(MSGPACK_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/msgpack)

FetchContent_Populate(download_msgpack
    GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
    GIT_TAG cpp-6.1.1
    SOURCE_DIR ${MSGPACK_INCLUDE_DIR}
)

add_definitions(-DMSGPACK_NO_BOOST)
add_definitions(-DMSGPACK_CXX17)

include_directories(${MSGPACK_INCLUDE_DIR}/include)
