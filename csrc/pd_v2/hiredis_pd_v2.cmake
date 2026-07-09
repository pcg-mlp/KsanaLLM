# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
#
# hiredis (Redis C client) for pd_v2.
#
# pd_v2 depends on hiredis in two places:
#   1. The vendored Mooncake tent backend's runtime
#      (mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp)
#      includes <hiredis/hiredis.h> *unconditionally* (it is compiled into
#      tent_runtime, which always builds), so the header must always be on
#      the include path — independent of whether the redis metastore is on.
#   2. csrc/pd_v2/mooncake/redis_alive_store.cpp uses the raw hiredis sync
#      API directly.
#
# Upstream Mooncake expects hiredis via the system package manager
# (`apt install libhiredis-dev`, dependencies.sh). That is not available on
# all target hosts (e.g. TencentOS, where neither the header nor an
# rpm exists). To keep the pd_v2 build self-contained — matching the
# vendored-FetchContent pattern in cmake/external/mooncake_tent.cmake — we
# fetch and build hiredis here and expose its include + lib roots.
#
# Built as a STATIC PIC archive: it gets bundled into the
# pd_v2_connector_mooncake.so along with the other static archives, so a
# static lib avoids adding a runtime .so dependency.
#
# Included by csrc/CMakeLists.txt (by explicit path) BEFORE
# external/mooncake_tent so the PD_V2_HIREDIS_* vars below are available to
# the tent child-build's configure step.

include(FetchContent)

# Pinned for reproducible builds. Bump + delete the source dir to refresh.
if(NOT DEFINED PD_V2_HIREDIS_REF)
  set(PD_V2_HIREDIS_REF "v1.2.0")
endif()

set(PD_V2_HIREDIS_SOURCE_DIR  ${THIRD_PARTY_PATH}/install/hiredis_pd_v2)
set(PD_V2_HIREDIS_BUILD_DIR   ${THIRD_PARTY_PATH}/build/hiredis_pd_v2)
set(PD_V2_HIREDIS_PREFIX      ${THIRD_PARTY_PATH}/install/hiredis_pd_v2_prefix)

FetchContent_Populate(download_hiredis_pd_v2
    GIT_REPOSITORY https://github.com/redis/hiredis.git
    GIT_TAG ${PD_V2_HIREDIS_REF}
    GIT_SHALLOW TRUE
    # Skip update on subsequent cmake runs — same rationale as
    # mooncake_tent.cmake: the FetchContent default re-runs git fetch every
    # configure step (slow, and fails behind proxies). Once cloned the source
    # is stable; bump PD_V2_HIREDIS_REF + delete the source dir to refresh.
    UPDATE_DISCONNECTED TRUE
    SOURCE_DIR ${PD_V2_HIREDIS_SOURCE_DIR}
)

# Stamp file so we don't re-configure/build hiredis on every cmake run. Bump
# the suffix to force a clean rebuild (e.g. after a GIT_TAG change).
set(_pd_v2_hiredis_stamp "${PD_V2_HIREDIS_BUILD_DIR}/.hiredis_pd_v2_stamp_v1")

if(NOT EXISTS "${_pd_v2_hiredis_stamp}")
  message(STATUS "Building vendored hiredis (${PD_V2_HIREDIS_REF}) at ${PD_V2_HIREDIS_BUILD_DIR}")
  file(MAKE_DIRECTORY ${PD_V2_HIREDIS_BUILD_DIR})

  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            # Static only: bundled into pd_v2_connector_mooncake.so, no
            # separate runtime .so to ship.
            -DBUILD_SHARED_LIBS=OFF
            -DDISABLE_TESTS=ON
            -DENABLE_SSL=OFF
            -DCMAKE_INSTALL_PREFIX=${PD_V2_HIREDIS_PREFIX}
            -S ${PD_V2_HIREDIS_SOURCE_DIR}
            -B ${PD_V2_HIREDIS_BUILD_DIR}
    RESULT_VARIABLE _pd_v2_hiredis_cfg_rc)
  if(NOT _pd_v2_hiredis_cfg_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 hiredis configure failed (exit ${_pd_v2_hiredis_cfg_rc})")
  endif()

  include(ProcessorCount)
  ProcessorCount(_pd_v2_hiredis_nproc)
  if(_pd_v2_hiredis_nproc EQUAL 0)
    set(_pd_v2_hiredis_nproc 8)
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${PD_V2_HIREDIS_BUILD_DIR}
            --target install --parallel ${_pd_v2_hiredis_nproc}
    RESULT_VARIABLE _pd_v2_hiredis_bld_rc)
  if(NOT _pd_v2_hiredis_bld_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 hiredis build/install failed (exit ${_pd_v2_hiredis_bld_rc})")
  endif()

  file(WRITE "${_pd_v2_hiredis_stamp}" "ok")
endif()

# Resolve the installed static archive. hiredis installs to lib or lib64
# depending on the platform; check both.
find_library(PD_V2_HIREDIS_LIB
    NAMES hiredis libhiredis
    HINTS ${PD_V2_HIREDIS_PREFIX}/lib ${PD_V2_HIREDIS_PREFIX}/lib64
    NO_DEFAULT_PATH)
if(NOT PD_V2_HIREDIS_LIB)
  message(FATAL_ERROR
    "pd_v2 hiredis: libhiredis.a not found under ${PD_V2_HIREDIS_PREFIX} "
    "(delete ${PD_V2_HIREDIS_BUILD_DIR} to force a rebuild)")
endif()

set(PD_V2_HIREDIS_INCLUDE_DIR ${PD_V2_HIREDIS_PREFIX}/include)

message(STATUS "pd_v2 hiredis include: ${PD_V2_HIREDIS_INCLUDE_DIR}")
message(STATUS "pd_v2 hiredis lib:     ${PD_V2_HIREDIS_LIB}")
