# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# pd_v2 real Mooncake backend: libpd_v2_connector_mooncake.so.
#
# Layout:
#   engine_tent_wrapper.{h,cpp}       wraps mooncake::TransferEngine engine-side
#   executor_tent_wrapper.{h,cpp}     wraps mooncake::TransferEngine executor-side
#   host_stage_pool.{h,cpp}           Prefill-side host stage buffer for the
#                                     control-plane prefill_req path (see
#                                     docs/technology/design/pd_v2_control_stage.md)
#   mooncake_engine_connector.{h,cpp} PdV2ConnectorInterface engine-side impl
#   mooncake_executor_connector.{h,cpp} PdV2ConnectorInterface executor-side impl
#   factory.cpp                       extern "C" ksana_pd_v2_create_connector
#
# The wrappers call the real Mooncake API (TransferEngine::init,
# installTransport, registerLocalMemory, openSegment, submitTransfer,
# sendNotifyByName, getNotifies). The .so is dlopen'd at runtime by
# ksana_engine / ksana_executor — engine/ and executor/ only link
# pd_v2_public + libdl, not Mooncake itself.

# Shared library: dlopen'd at runtime by ksana_engine / ksana_executor
# bootstrap.
add_library(pd_v2_connector_mooncake SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_tent_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/executor_tent_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host_stage_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/peer_credits.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_engine_connector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_engine_connector_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_engine_connector_prefill.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_engine_connector_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_executor_connector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/redis_alive_store.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/factory.cpp
)
target_include_directories(pd_v2_connector_mooncake PUBLIC ${PROJECT_SOURCE_DIR}/csrc)
# hiredis ships under /usr/local on the dev/CI hosts that already build
# Mooncake (its own CMake locates the same library via find_library).
# pd_v2/decode_alive/<cluster>/* uses raw hiredis (sync API) directly rather than
# routing through Mooncake, so we explicitly add the include path.
target_include_directories(pd_v2_connector_mooncake PRIVATE /usr/local/include)
target_link_libraries(pd_v2_connector_mooncake
    PUBLIC pd_v2_public pd_v2_wire pd_v2_decode utils)
target_link_libraries(pd_v2_connector_mooncake PRIVATE pthread)

# Real Mooncake link: include path + static archives.
# Headers transitively pull in glog; GLOG_USE_GLOG_EXPORT switches to
# the modern symbol export mode used by upstream Mooncake (matches
# csrc/global_cache_connector/CMakeLists.txt). Mooncake headers also
# use C++20 std::string::starts_with, so the CXX standard is bumped.
# Build artifacts come from cmake/external/mooncake_tent.cmake which
# fetches the Mooncake tent-backend at configure time and builds only
# the standalone mooncake-transfer-engine subtree (no store/master).
target_include_directories(pd_v2_connector_mooncake PRIVATE
    ${PD_V2_MOONCAKE_INCLUDE_DIRS})
target_compile_definitions(pd_v2_connector_mooncake PRIVATE GLOG_USE_GLOG_EXPORT USE_TENT)
set_target_properties(pd_v2_connector_mooncake PROPERTIES CXX_STANDARD 20)
set_property(TARGET pd_v2_connector_mooncake APPEND PROPERTY
    BUILD_RPATH ${PD_V2_MOONCAKE_BUILD_DIR}/mooncake-common)
target_link_libraries(pd_v2_connector_mooncake PRIVATE
    -Wl,--whole-archive
    ${PD_V2_MOONCAKE_BUILD_DIR}/src/libtransfer_engine.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/src/common/base/libbase.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/mooncake-common-src/libmooncake_common.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/libtent.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/platform/cuda/libplatform_cuda.a
    -Wl,--no-whole-archive
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/metrics/libtent_metrics.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/platform/libtent_platform_all.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/transport/tcp/libtent_xport_tcp.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/transport/rdma/libtent_xport_rdma.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/transport/bufio/libtent_xport_bufio.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/transport/shm/libtent_xport_shm.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/transport/nvlink/libtent_xport_nvlink.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/transport/mnnvl/libtent_xport_mnnvl.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/transport/gds/libtent_xport_gds.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/metastore/libmetastore_redis.a
    ${PD_V2_MOONCAKE_BUILD_DIR}/tent/src/metastore/libmetastore_http.a
    ${PD_V2_MOONCAKE_DEPS_PREFIX}/lib/libglog.a
    ${PD_V2_MOONCAKE_DEPS_PREFIX}/lib/libjsoncpp.a
    ${PD_V2_MOONCAKE_DEPS_PREFIX}/lib/libyaml-cpp.a
    ${PD_V2_MOONCAKE_DEPS_PREFIX}/lib/libhiredis.a
    gflags numa ibverbs rdmacm cudart
    ${PD_V2_MOONCAKE_BUILD_DIR}/mooncake-common/libasio.so)

# This .so is a self-contained dlopen plugin: inference_engine loads it with
# RTLD_GLOBAL, and it statically vendors its own gflags/glog/mooncake. Without
# this, gflags' global flag registry (FLAGS_flagfile, ...) is exported from
# BOTH this .so and torch_serving.so (which also links gflags); RTLD_GLOBAL
# merges them into one namespace, gflags sees `flagfile` registered twice, and
# the process aborts at startup ("flag 'flagfile' ... being linked both
# statically and dynamically"). --exclude-libs,ALL localizes every symbol that
# comes from a static archive (gflags/glog/mooncake/utils/...), so the vendored
# gflags stays private to this plugin. The only symbol that must remain exported
# is the factory entry point ksana_pd_v2_create_connector — it lives in
# factory.cpp (an object file, not an archive) and is therefore unaffected.
target_link_options(pd_v2_connector_mooncake PRIVATE -Wl,--exclude-libs,ALL)

if(WITH_STANDALONE_TEST)
    # RedisAliveStore unit test. Skips at runtime when the smoke Redis
    # is unreachable so non-networked CI doesn't block. Compiles
    # redis_alive_store.cpp directly (rather than depending on the
    # connector .so) to keep the test isolated from Mooncake link.
    # hiredis is vendored into the pd_v2 private deps prefix by
    # cmake/external/mooncake_tent.cmake (not host-installed under
    # /usr/local on lean CI images), so link the static archive by full
    # path and add its include dir explicitly — same source as the
    # connector library above.
    cpp_test(pd_v2_redis_alive_store_test
        SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/redis_alive_store_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/redis_alive_store.cpp
        DEPS utils
        LIBS ${PD_V2_MOONCAKE_DEPS_PREFIX}/lib/libhiredis.a)
    # cpp_test gates on WITH_TESTING; only patch the target when it
    # actually got created.
    if(TARGET pd_v2_redis_alive_store_test)
        target_include_directories(pd_v2_redis_alive_store_test PRIVATE
            ${PROJECT_SOURCE_DIR}/csrc
            ${PD_V2_MOONCAKE_DEPS_PREFIX}/include
            /usr/local/include)
    endif()

    # Cross-device copy contract test. Pins down the CUDA-runtime premises the
    # vendored Mooncake patch depends on. Skips at runtime if fewer than 2 GPUs
    # are visible. Doesn't link Mooncake; it's a pure cuda_runtime contract test.
    cpp_test(pd_v2_cross_device_copy_contract_test
        SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/k8a_cross_device_copy_test.cpp
        DEPS utils
        LIBS cudart)
    if(TARGET pd_v2_k8a_cross_device_copy_test)
        target_include_directories(pd_v2_k8a_cross_device_copy_test PRIVATE
            ${PROJECT_SOURCE_DIR}/csrc)
    endif()

    # HostStagePool unit test. Pure C++/STL + mmap; no Mooncake link, so
    # it compiles host_stage_pool.cpp directly rather than depending on
    # the connector .so. Small 16 MiB pool keeps the test under default
    # ulimit -l on dev hosts (mlock failure is logged but non-fatal).
    cpp_test(pd_v2_host_stage_pool_test
        SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/host_stage_pool_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/host_stage_pool.cpp
        DEPS utils)
    if(TARGET pd_v2_host_stage_pool_test)
        target_include_directories(pd_v2_host_stage_pool_test PRIVATE
            ${PROJECT_SOURCE_DIR}/csrc)
    endif()

    # PeerCredits unit test. Pure C++/STL; no Mooncake link, compiles
    # peer_credits.cpp standalone same as host_stage_pool_test.
    cpp_test(pd_v2_peer_credits_test
        SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/peer_credits_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/peer_credits.cpp
        DEPS utils)
    if(TARGET pd_v2_peer_credits_test)
        target_include_directories(pd_v2_peer_credits_test PRIVATE
            ${PROJECT_SOURCE_DIR}/csrc)
    endif()

    # ===== Fake-tent coverage tests =====
    # Compile the REAL mooncake tent wrappers (and, in the connector test,
    # the real connectors) against a FAKE in-process mooncake::TransferEngine
    # (fake_tent/) instead of the real Mooncake static archives. This
    # exercises the connector/wrapper .cpp for coverage without RDMA, a
    # metadata server, or a second process. The fake_tent include dir is
    # added BEFORE everything so its transfer_engine.h / transport/transport.h
    # / transfer_metadata.h / common/base/status.h shadow the real headers;
    # the real Mooncake include dirs are deliberately NOT on the path.
    cpp_test(pd_v2_tent_wrapper_fake_test
        SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_tent_wrapper_fake_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/engine_tent_wrapper.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/executor_tent_wrapper.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/fake_tent/fake_tent_engine.cpp
        DEPS pd_v2_public pd_v2_wire utils)
    if(TARGET pd_v2_tent_wrapper_fake_test)
        target_include_directories(pd_v2_tent_wrapper_fake_test BEFORE PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/fake_tent
            ${PROJECT_SOURCE_DIR}/csrc
            ${PROJECT_BINARY_DIR}/third_party/install/base64/include
            ${PROJECT_BINARY_DIR}/third_party/install/nlohmann_json/single_include
            ${PROJECT_BINARY_DIR}/third_party/download/fmt/include)
        target_compile_definitions(pd_v2_tent_wrapper_fake_test PRIVATE USE_TENT)
    endif()

    # MooncakeExecutorConnector against the fake tent + fake IpcFuncWrapper.
    # Compiles the real connector + executor wrapper + redis_alive_store
    # (linked only — epoch poll uses P2PHANDSHAKE metadata so it never runs).
    cpp_test(pd_v2_executor_connector_fake_test
        SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_executor_connector_fake_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_executor_connector.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/executor_tent_wrapper.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/redis_alive_store.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/fake_tent/fake_tent_engine.cpp
        DEPS pd_v2_public pd_v2_wire utils device
        LIBS ${PD_V2_MOONCAKE_DEPS_PREFIX}/lib/libhiredis.a)
    if(TARGET pd_v2_executor_connector_fake_test)
        target_include_directories(pd_v2_executor_connector_fake_test BEFORE PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/fake_tent
            ${PROJECT_SOURCE_DIR}/csrc
            ${PROJECT_BINARY_DIR}/third_party/install/base64/include
            ${PROJECT_BINARY_DIR}/third_party/install/nlohmann_json/single_include
            ${PROJECT_BINARY_DIR}/third_party/download/fmt/include
            ${PD_V2_MOONCAKE_DEPS_PREFIX}/include
            /usr/local/include)
        target_compile_definitions(pd_v2_executor_connector_fake_test PRIVATE USE_TENT)
    endif()

    # MooncakeEngineConnector against the fake tent + fake IpcFuncWrapper.
    # Pulls peer_selector / linucb (pd_v2_decode), host_stage_pool, peer_credits,
    # redis_alive_store. The Redis discovery case skips at runtime when Redis
    # is unreachable; the rest run over P2PHANDSHAKE with no external deps.
    cpp_test(pd_v2_engine_connector_fake_test
        SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_engine_connector_fake_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_engine_connector.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_engine_connector_decode.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_engine_connector_prefill.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mooncake_engine_connector_discovery.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/engine_tent_wrapper.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/host_stage_pool.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/peer_credits.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/redis_alive_store.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/fake_tent/fake_tent_engine.cpp
        DEPS pd_v2_public pd_v2_wire pd_v2_decode utils device
        LIBS ${PD_V2_MOONCAKE_DEPS_PREFIX}/lib/libhiredis.a)
    if(TARGET pd_v2_engine_connector_fake_test)
        target_include_directories(pd_v2_engine_connector_fake_test BEFORE PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/fake_tent
            ${PROJECT_SOURCE_DIR}/csrc
            ${PROJECT_BINARY_DIR}/third_party/install/base64/include
            ${PROJECT_BINARY_DIR}/third_party/install/nlohmann_json/single_include
            ${PROJECT_BINARY_DIR}/third_party/download/fmt/include
            ${PD_V2_MOONCAKE_DEPS_PREFIX}/include
            /usr/local/include)
        target_compile_definitions(pd_v2_engine_connector_fake_test PRIVATE USE_TENT)
    endif()
endif()
