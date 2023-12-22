# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(NLOHMANN_JSON_SRC_DIR ${THIRD_PARTY_PATH}/install/nlohmann_json)

FetchContent_Declare(json
    GIT_REPOSITORY https://git.woa.com/cpp_thirdparty/nlohmann_json.git
    GIT_TAG  bc889afb4c5bf1c0d8ee29ef35eaaf4c8bef8a5d              # v3.11.2
    SOURCE_DIR ${NLOHMANN_JSON_SRC_DIR}
)

FetchContent_MakeAvailable(json)

include_directories(${NLOHMANN_JSON_SRC_DIR}/single_include)