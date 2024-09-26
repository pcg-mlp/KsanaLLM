# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party")

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

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/tests
  ${THIRD_PARTY_PATH}/install/loguru
  ${THIRD_PARTY_PATH}/pybind11/include
  ${THIRD_PARTY_PATH}/download/fmt/include
  ${Protobuf_INCLUDE_DIR}
  ${OPENTELEMETRY_CPP_INCLUDE_DIRS}
)

add_subdirectory(batch_manager)
add_subdirectory(batch_scheduler)
add_subdirectory(block_manager)
add_subdirectory(cache_manager)
add_subdirectory(endpoints)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(models)
add_subdirectory(plugins)
add_subdirectory(runtime)
add_subdirectory(samplers)
add_subdirectory(service)
add_subdirectory(profiler)
add_subdirectory(utils)
add_subdirectory(periphery)
add_subdirectory(cpp)

add_library(ksana_llm_libs INTERFACE)
target_link_libraries(ksana_llm_libs INTERFACE
  kernels
  layers
  models
  plugins
  service
  utils
  batch_manager
  batch_scheduler
  block_manager
  cache_manager
  runtime
  endpoints
  samplers
  profiler
  loguru
  gflags
  periphery
  ${NCCL_LIBRARIES}
  ${Protobuf_LIBRARIES}
)

add_subdirectory(torch_op)
