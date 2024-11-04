# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/tests
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
)

add_subdirectory(torch_op)
