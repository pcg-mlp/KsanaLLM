# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party")

# import external dependencies
include(external/protobuf)
include(external/fmt)
include(external/loguru)
include(external/gflags)
include(external/httplib)
include(external/pybind11)
include(external/yaml-cpp)
include(external/nlohmann_json)
include(external/base64)
include(external/msgpack)
include(external/opentelemetry)

add_subdirectory(ksana_llm)
add_subdirectory(endpoints_impl)
