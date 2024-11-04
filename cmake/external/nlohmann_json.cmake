# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(NLOHMANN_JSON_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/nlohmann_json)

FetchContent_Populate(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    SOURCE_DIR ${NLOHMANN_JSON_INCLUDE_DIR}
)

include_directories(${NLOHMANN_JSON_INCLUDE_DIR}/single_include)
