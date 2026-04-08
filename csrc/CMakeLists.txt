# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# import external dependencies
include(external/abseil-cpp)
include(external/protobuf)
include(external/tbb)
include(external/fmt)
include(external/loguru)
include(external/gflags)
include(external/httplib)
include(external/pybind11)
include(external/nlohmann_json)
include(external/base64)
include(external/msgpack)
include(external/re2)
include(external/opentelemetry)
include(external/polaris_cpp)
include(external/yaml-cpp)
include(external/cppzmq)
include(external/libzmq)
include(external/xgrammar)

include_directories(
  ${PROJECT_SOURCE_DIR}/csrc
  ${PROJECT_SOURCE_DIR}/tests
)

add_subdirectory(batch_scheduler)
add_subdirectory(cache_manager)
add_subdirectory(distributed)
add_subdirectory(endpoints)
add_subdirectory(executor)
add_subdirectory(ipc)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(modules)
add_subdirectory(model_loader)
add_subdirectory(models)
add_subdirectory(models/base)
add_subdirectory(runtime)
add_subdirectory(configure)
add_subdirectory(samplers)
add_subdirectory(service)
add_subdirectory(engine)
add_subdirectory(supervisor)
add_subdirectory(connector)
add_subdirectory(global_cache_connector)
add_subdirectory(transfer)
add_subdirectory(profiler)
add_subdirectory(device)
add_subdirectory(utils)
add_subdirectory(periphery)

add_library(ksana_llm_libs INTERFACE)
target_link_libraries(ksana_llm_libs INTERFACE
  model_loader
  models
  modules
  model_base
  layers
  service
  supervisor
  engine
  -Wl,--start-group
  executor
  ipc
  -Wl,--end-group
  utils
  connector
  global_cache_connector
  batch_scheduler
  cache_manager
  distributed
  runtime
  configure 
  endpoints
  samplers
  kernels
  profiler
  device 
  loguru
  gflags
  periphery
  ${NCCL_LIBRARIES}
)

add_subdirectory(torch_op)
