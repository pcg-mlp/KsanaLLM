# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

include(external/httplib)
include(external/base64)
include(external/msgpack)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
)

add_executable(serving_forward_client ${PROJECT_SOURCE_DIR}/src/ksana_llm/cpp/serving_forward_client.cpp)
