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
include(external/libcurl)
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

# pd_v2 vendored Mooncake transfer-engine (tent backend). Gated so
# CPU/NPU and non-pd_v2 builds don't pay the FetchContent + Mooncake
# build cost (~minutes), and offline / proxy-isolated environments
# stay green when pd_v2 isn't needed.
#
# The Mooncake tent sources require a recent glibc (>= 2.35): their headers
# transitively pull in glibc-internal fixed-width-int headers (e.g.
# <bits/stdint-uintn.h>, which only exists on glibc >= 2.25) and use rdma-core
# APIs that only ship on the newer toolchains paired with such glibc. On older
# systems (e.g. CentOS 7 / glibc 2.17) Mooncake simply will not compile.
#
# Because pd_v2 is an isolated directory wired in only through a dlopen'd
# C-ABI connector (libpd_v2_connector_mooncake.so) — nothing links it
# statically — skipping it costs the rest of the build NOTHING. So we DEFAULT
# the option OFF on old glibc (normal builds stay fully green), and only if the
# user *explicitly* requests it there do we fail loudly: it cannot work.
include(CheckCXXSourceCompiles)
set(_kllm_save_req_quiet "${CMAKE_REQUIRED_QUIET}")
set(CMAKE_REQUIRED_QUIET ON)
check_cxx_source_compiles(
  "#include <features.h>\n#if defined(__GLIBC__) && (__GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 35))\n#error glibc too old for pd_v2\n#endif\nint main(){ return 0; }"
  _KLLM_PD_V2_GLIBC_OK)
set(CMAKE_REQUIRED_QUIET "${_kllm_save_req_quiet}")

if(_KLLM_PD_V2_GLIBC_OK)
  set(_kllm_pd_v2_default ON)
else()
  set(_kllm_pd_v2_default OFF)
endif()

option(ENABLE_PD_V2_MOONCAKE
       "Build the vendored Mooncake transfer-engine (tent) needed by pd_v2 (requires glibc >= 2.35)."
       ${_kllm_pd_v2_default})

if(NOT _KLLM_PD_V2_GLIBC_OK)
  if(ENABLE_PD_V2_MOONCAKE)
    message(FATAL_ERROR
      "ENABLE_PD_V2_MOONCAKE=ON was requested, but the pd_v2 Mooncake backend "
      "requires glibc >= 2.35 and this host's glibc is older. pd_v2 is "
      "unavailable here. The rest of KsanaLLM builds fine without it: re-run "
      "cmake with -DENABLE_PD_V2_MOONCAKE=OFF (the default on this system), or "
      "build on a host with glibc >= 2.35 to use pd_v2.")
  else()
    message(STATUS
      "pd_v2 Mooncake backend disabled: requires glibc >= 2.35 and this host's "
      "glibc is older. The rest of KsanaLLM builds normally.")
  endif()
endif()

if(ENABLE_PD_V2_MOONCAKE)
  include(external/mooncake_tent)
endif()

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
add_subdirectory(pd_v2)
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
  -Wl,--whole-archive
  ipc
  -Wl,--no-whole-archive
  -Wl,--end-group
  utils
  connector
  global_cache_connector
  pd_v2
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
  ${XPU_SHARED_LIBS}
)

add_subdirectory(torch_op)

# LLM_kernels 远程缓存 upload：挂到 torch_serving，使 make torch_serving 也会上传
if(LLM_KERNELS_REMOTE_CACHE_UPLOAD_ENABLED AND TARGET llm_kernels_remote_cache_upload AND TARGET torch_serving)
  add_dependencies(torch_serving llm_kernels_remote_cache_upload)
endif()

# HIT 替换：先 stub 出 .a，再 apply_prebuilt 覆盖；链接必须等覆盖完成
if(TARGET llm_kernels_cache_apply_prebuilt AND TARGET torch_serving)
  add_dependencies(torch_serving llm_kernels_cache_apply_prebuilt)
endif()
