# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(PYBIND11_INSTALL_DIR ${THIRD_PARTY_PATH}/pybind11)

FetchContent_Populate(download_pybind11
    GIT_REPOSITORY https://git.woa.com/cpp_thirdparty/pybind11.git
    GIT_TAG ffa346860b306c9bbfb341aed9c14c067751feb8
    SOURCE_DIR ${PYBIND11_INSTALL_DIR}
    SUBBUILD_DIR ${THIRD_PARTY_PATH}/tmp
    BINARY_DIR ${THIRD_PARTY_PATH}/tmp
)

include_directories(${PYBIND11_INSTALL_DIR}/include)
