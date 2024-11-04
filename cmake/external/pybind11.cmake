# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(PYBIND11_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/pybind11)

FetchContent_Populate(download_pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.5
    SOURCE_DIR ${PYBIND11_INCLUDE_DIR}
)

include_directories(${PYBIND11_INCLUDE_DIR}/include)
