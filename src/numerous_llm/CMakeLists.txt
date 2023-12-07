# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party")

include(external/fmt)
include(external/loguru)
include(external/gflags)
include(external/httplib)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${THIRD_PARTY_PATH}/install/loguru
  ${THIRD_PARTY_PATH}/download/fmt/include
)

add_subdirectory(batch_manager)
add_subdirectory(block_manager)
add_subdirectory(endpoints)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(models)
add_subdirectory(plugins)
add_subdirectory(runtime)
add_subdirectory(samplers)
add_subdirectory(service)
add_subdirectory(torch_op)
add_subdirectory(utils)

add_executable(numerous_llm main.cpp)
target_link_libraries(numerous_llm PUBLIC 
  -lpthread -ldl
  kernels
  layers
  models
  plugins
  service
  torch_op
  utils
  batch_manager 
  block_manager
  runtime
  endpoints 
  samplers
  loguru
  gflags
)
