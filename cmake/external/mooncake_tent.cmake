# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
#
# Mooncake transfer-engine (tent backend) for pd_v2.
#
# Separate from cmake/external/mooncake.cmake (which pins v0.3.9 for the
# legacy global_cache_connector). pd_v2 needs the tent backend, which
# v0.3.9 doesn't fully expose; we pull a recent Mooncake and build only
# the `mooncake-transfer-engine` subdir (its CMakeLists supports a
# standalone build via the `if (NOT GLOBAL_CONFIG)` branch).
#
# Why a separate Mooncake build:
#   - The full Mooncake monorepo builds mooncake_client (in mooncake-store)
#     which references TransferEngine::getBaseAddr() — an API the tent
#     backend doesn't implement, so the full build fails at link time.
#   - pd_v2 only needs transfer-engine + tent. The transfer-engine
#     subdir's CMake supports standalone build.
#
# TODO(pd_v2): pin to a release tag once tent stabilizes. Currently using
# main while waiting for upstream. PD_V2_MOONCAKE_REF can be set on the
# cmake command line (-DPD_V2_MOONCAKE_REF=<sha-or-tag>) to override.

include(FetchContent)

# Default ref. Pinned to a known-good commit for reproducible builds and
# so that `cmake/patches/*.patch` apply cleanly. CMake user can override
# via `-DPD_V2_MOONCAKE_REF=<sha-or-tag>` on the command line; if you
# bump this, re-verify all patches in cmake/patches/ still apply
# (`git apply --check`) and re-run smoke.
#
# Pin history:
#   - 2026-05-17: 368b41f1 (main HEAD as of K8a CudaPlatform::copy patch)
#   - 2026-06-09: v0.3.11 (b4ccdc30) — upgrade to pick up #2102 (da959168)
#       which refactors TENT endpoint lifecycle to single-direction
#       (UNINIT→HANDSHAKING→READY→DESTROYING→DESTROYED), deprecates
#       reset(), and removes the stale notify_qp_ handling in
#       resetUnlocked() that was poisoning the notify QP state after
#       any endpoint reuse — symptom: large stage WRITE (~194 KB)
#       caused subsequent SendNotification to deliver empty payload
#       (name='', msg_size=0) on the receiver side.
if(NOT DEFINED PD_V2_MOONCAKE_REF)
  set(PD_V2_MOONCAKE_REF "b4ccdc3082d2def865ecf1b5e68bc511025da40e")
endif()

# Respect the project-level CUDA toggle. Mooncake tent has both CUDA
# and non-CUDA build paths; previously this file forced USE_CUDA=ON
# unconditionally even when the parent project targeted CPU-only
# (review §P1.2).
if(WITH_CUDA)
  set(_PD_V2_MOONCAKE_USE_CUDA ON)
else()
  set(_PD_V2_MOONCAKE_USE_CUDA OFF)
endif()

set(PD_V2_MOONCAKE_SOURCE_DIR ${THIRD_PARTY_PATH}/install/mooncake_pd_v2)
set(PD_V2_MOONCAKE_BUILD_DIR  ${THIRD_PARTY_PATH}/build/mooncake_pd_v2)

FetchContent_Populate(download_mooncake_pd_v2
    GIT_REPOSITORY https://github.com/kvcache-ai/Mooncake.git
    GIT_TAG ${PD_V2_MOONCAKE_REF}
    GIT_SHALLOW TRUE
    # Skip update on subsequent cmake runs — the FetchContent default re-runs
    # git fetch every configure step, which (a) is slow and (b) fails behind
    # corporate proxies that don't get inherited. Once cloned, the source dir
    # is stable; bump PD_V2_MOONCAKE_REF + delete the source dir to refresh.
    UPDATE_DISCONNECTED TRUE
    SOURCE_DIR ${PD_V2_MOONCAKE_SOURCE_DIR}
)

# Apply local Mooncake patches (idempotent — `git apply --check` succeeds
# only when none of the diff is currently applied; if it's already in,
# we skip). One patch per file in cmake/patches/, applied in name order.
# Must run BEFORE the build stamp check below so a re-build picks up
# patch changes.
file(GLOB _pd_v2_mc_patches LIST_DIRECTORIES false
     "${PROJECT_SOURCE_DIR}/cmake/patches/*.patch")
list(SORT _pd_v2_mc_patches)
foreach(_patch IN LISTS _pd_v2_mc_patches)
  execute_process(
    COMMAND git -C ${PD_V2_MOONCAKE_SOURCE_DIR} apply --check -- ${_patch}
    RESULT_VARIABLE _check_rc
    OUTPUT_QUIET ERROR_QUIET)
  if(_check_rc EQUAL 0)
    message(STATUS "Applying Mooncake patch: ${_patch}")
    execute_process(
      COMMAND git -C ${PD_V2_MOONCAKE_SOURCE_DIR} apply -- ${_patch}
      RESULT_VARIABLE _apply_rc)
    if(NOT _apply_rc EQUAL 0)
      message(FATAL_ERROR "Failed to apply Mooncake patch: ${_patch} (exit ${_apply_rc})")
    endif()
    # Force rebuild so the patched source rebuilds libtransfer_engine.a.
    file(REMOVE "${PD_V2_MOONCAKE_BUILD_DIR}/.pd_v2_stamp_v1")
  else()
    message(STATUS "Mooncake patch already applied (or conflicts): ${_patch}")
  endif()
endforeach()

# Upstream TENT currently adds its tests unconditionally from
# mooncake-transfer-engine/tent/CMakeLists.txt. The pd_v2 vendored build only
# needs the libraries; building tests pulls gtest into the sub-build and fails
# on lean CI images. Gate the tests locally until upstream makes it conditional.
set(_pd_v2_tent_cmake "${PD_V2_MOONCAKE_SOURCE_DIR}/mooncake-transfer-engine/tent/CMakeLists.txt")
file(READ "${_pd_v2_tent_cmake}" _pd_v2_tent_cmake_content)
string(REPLACE
       "add_subdirectory(tests)"
       "if(BUILD_TENT_TESTS)\n  add_subdirectory(tests)\nendif()"
       _pd_v2_tent_cmake_patched
       "${_pd_v2_tent_cmake_content}")
if(NOT _pd_v2_tent_cmake_patched STREQUAL _pd_v2_tent_cmake_content)
  message(STATUS "Patching Mooncake TENT tests behind BUILD_TENT_TESTS")
  file(WRITE "${_pd_v2_tent_cmake}" "${_pd_v2_tent_cmake_patched}")
  file(REMOVE "${PD_V2_MOONCAKE_BUILD_DIR}/.pd_v2_stamp_v1")
endif()

# Provide config-mode dependencies to the Mooncake tent sub-build.
#
# Mooncake's mooncake-common/common.cmake does `find_package(yaml-cpp
# REQUIRED)` and `find_package(gflags REQUIRED)` (true for both v0.3.9 and
# current main). The tent sub-build below runs as a *separate* cmake
# process via execute_process, so it can't see KsanaLLM's
# `add_subdirectory`'d yaml-cpp / gflags targets — those paths build the
# libs but never install a `*-config.cmake` the child process can discover.
#
# Rather than depend on host-installed -dev packages (which some CI images
# lack — the original "Could not find a package configuration file" errors),
# build + install the project's already-fetched vendored sources into a
# private prefix and hand it to the sub-build via CMAKE_PREFIX_PATH.
# CMAKE_PREFIX_PATH is additive to the default system prefixes, so
# host-provided deps (e.g. yalantinglibs) continue to resolve normally.
#
# Source dirs come from the exporting cmake files (external/yaml-cpp.cmake,
# external/gflags.cmake), included before this file in csrc/CMakeLists.txt.
# We can't reconstruct the paths locally because the FetchContent content
# names may be populated elsewhere first (e.g. 3rdparty/LLM_kernels).
set(_PD_V2_DEPS_PREFIX ${THIRD_PARTY_PATH}/install/mooncake_pd_v2_deps)

# yaml-cpp (always built from source by external/yaml-cpp.cmake).
if(NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib/cmake/yaml-cpp/yaml-cpp-config.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib64/cmake/yaml-cpp/yaml-cpp-config.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/share/cmake/yaml-cpp/yaml-cpp-config.cmake")
  if(NOT EXISTS "${KSANA_YAML_CPP_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "pd_v2 yaml-cpp source not found at ${KSANA_YAML_CPP_SOURCE_DIR}. "
      "Expected external/yaml-cpp.cmake to run (and export "
      "KSANA_YAML_CPP_SOURCE_DIR) before external/mooncake_tent "
      "(see csrc/CMakeLists.txt).")
  endif()
  message(STATUS "Installing vendored yaml-cpp for Mooncake tent sub-build at ${_PD_V2_DEPS_PREFIX}")
  set(_pd_v2_yaml_bld ${PD_V2_MOONCAKE_BUILD_DIR}/yaml-cpp-build)
  file(MAKE_DIRECTORY ${_pd_v2_yaml_bld})
  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DYAML_CPP_BUILD_TESTS=OFF
            -DYAML_CPP_BUILD_TOOLS=OFF
            -DYAML_CPP_BUILD_CONTRIB=OFF
            -DCMAKE_INSTALL_PREFIX=${_PD_V2_DEPS_PREFIX}
            -DCMAKE_INSTALL_LIBDIR=lib
            -S ${KSANA_YAML_CPP_SOURCE_DIR}
            -B ${_pd_v2_yaml_bld}
    RESULT_VARIABLE _pd_v2_yaml_cfg_rc
    OUTPUT_VARIABLE _pd_v2_yaml_cfg_out
    ERROR_VARIABLE _pd_v2_yaml_cfg_err)
  if(NOT _pd_v2_yaml_cfg_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 yaml-cpp configure failed (exit ${_pd_v2_yaml_cfg_rc})\nCONFIGURE OUTPUT:\n${_pd_v2_yaml_cfg_out}\nCONFIGURE ERROR:\n${_pd_v2_yaml_cfg_err}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${_pd_v2_yaml_bld} --target install
    RESULT_VARIABLE _pd_v2_yaml_bld_rc
    OUTPUT_VARIABLE _pd_v2_yaml_bld_out
    ERROR_VARIABLE _pd_v2_yaml_bld_err)
  if(NOT _pd_v2_yaml_bld_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 yaml-cpp build/install failed (exit ${_pd_v2_yaml_bld_rc})\nBUILD OUTPUT:\n${_pd_v2_yaml_bld_out}\nBUILD ERROR:\n${_pd_v2_yaml_bld_err}")
  endif()
endif()

# gflags. external/gflags.cmake prefers system gflags and only sets
# KSANA_GFLAGS_SOURCE_DIR when it had to build from source. When system
# gflags is used the sub-build resolves it the same way, so we skip.
if(KSANA_GFLAGS_SOURCE_DIR
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib/cmake/gflags/gflags-config.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib64/cmake/gflags/gflags-config.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/share/cmake/gflags/gflags-config.cmake")
  if(NOT EXISTS "${KSANA_GFLAGS_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "pd_v2 gflags source not found at ${KSANA_GFLAGS_SOURCE_DIR}. "
      "Expected external/gflags.cmake to run (and export "
      "KSANA_GFLAGS_SOURCE_DIR) before external/mooncake_tent "
      "(see csrc/CMakeLists.txt).")
  endif()
  message(STATUS "Installing vendored gflags for Mooncake tent sub-build at ${_PD_V2_DEPS_PREFIX}")
  set(_pd_v2_gflags_bld ${PD_V2_MOONCAKE_BUILD_DIR}/gflags-build)
  file(MAKE_DIRECTORY ${_pd_v2_gflags_bld})
  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_STATIC_LIBS=ON
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_TESTING=OFF
            -DGFLAGS_NAMESPACE=google
            # Do NOT register this private install into CMake's user package
            # registry (~/.cmake/packages/gflags/). gflags defaults
            # REGISTER_INSTALL_PREFIX=ON, which makes its `install` step write a
            # registry entry pointing at this build's mooncake_pd_v2_deps prefix.
            # That registry is global to the user, so later unrelated builds'
            # `find_package(gflags)` would silently resolve to this (often
            # stale) workspace path and pull in a mismatched IMPORTED target.
            -DREGISTER_INSTALL_PREFIX=OFF
            -DREGISTER_BUILD_DIR=OFF
            -DCMAKE_INSTALL_PREFIX=${_PD_V2_DEPS_PREFIX}
            -DCMAKE_INSTALL_LIBDIR=lib
            -S ${KSANA_GFLAGS_SOURCE_DIR}
            -B ${_pd_v2_gflags_bld}
    RESULT_VARIABLE _pd_v2_gflags_cfg_rc
    OUTPUT_VARIABLE _pd_v2_gflags_cfg_out
    ERROR_VARIABLE _pd_v2_gflags_cfg_err)
  if(NOT _pd_v2_gflags_cfg_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 gflags configure failed (exit ${_pd_v2_gflags_cfg_rc})\nCONFIGURE OUTPUT:\n${_pd_v2_gflags_cfg_out}\nCONFIGURE ERROR:\n${_pd_v2_gflags_cfg_err}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${_pd_v2_gflags_bld} --target install
    RESULT_VARIABLE _pd_v2_gflags_bld_rc
    OUTPUT_VARIABLE _pd_v2_gflags_bld_out
    ERROR_VARIABLE _pd_v2_gflags_bld_err)
  if(NOT _pd_v2_gflags_bld_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 gflags build/install failed (exit ${_pd_v2_gflags_bld_rc})\nBUILD OUTPUT:\n${_pd_v2_gflags_bld_out}\nBUILD ERROR:\n${_pd_v2_gflags_bld_err}")
  endif()
endif()

# glog. Mooncake links targets against glog::glog. Keep gflags support enabled:
# upstream Mooncake expects a normal glog build, and gflags is already provided
# in the same private prefix for the tent sub-build.
set(_PD_V2_GLOG_REF "v0.6.0")
set(_PD_V2_GLOG_SRC ${THIRD_PARTY_PATH}/install/glog_pd_v2)
if(NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib/cmake/glog/glog-config.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib64/cmake/glog/glog-config.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/share/cmake/glog/glog-config.cmake")
  if(NOT EXISTS "${_PD_V2_GLOG_SRC}/CMakeLists.txt")
    message(STATUS "Fetching glog ${_PD_V2_GLOG_REF} for Mooncake tent sub-build")
    FetchContent_Populate(download_glog_pd_v2
        GIT_REPOSITORY https://github.com/google/glog.git
        GIT_TAG ${_PD_V2_GLOG_REF}
        GIT_SHALLOW TRUE
        UPDATE_DISCONNECTED TRUE
        SOURCE_DIR ${_PD_V2_GLOG_SRC})
  endif()
  message(STATUS "Installing glog ${_PD_V2_GLOG_REF} for Mooncake tent sub-build at ${_PD_V2_DEPS_PREFIX}")
  set(_pd_v2_glog_bld ${PD_V2_MOONCAKE_BUILD_DIR}/glog-build)
  file(MAKE_DIRECTORY ${_pd_v2_glog_bld})
  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_TESTING=OFF
            -DWITH_GFLAGS=ON
            -DWITH_GTEST=OFF
            -DWITH_UNWIND=OFF
            -DCMAKE_PREFIX_PATH=${_PD_V2_DEPS_PREFIX}
            -DCMAKE_CXX_FLAGS=-I${_PD_V2_DEPS_PREFIX}/include
            -DCMAKE_INSTALL_PREFIX=${_PD_V2_DEPS_PREFIX}
            -DCMAKE_INSTALL_LIBDIR=lib
            -S ${_PD_V2_GLOG_SRC}
            -B ${_pd_v2_glog_bld}
    RESULT_VARIABLE _pd_v2_glog_cfg_rc
    OUTPUT_VARIABLE _pd_v2_glog_cfg_out
    ERROR_VARIABLE _pd_v2_glog_cfg_err)
  if(NOT _pd_v2_glog_cfg_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 glog configure failed (exit ${_pd_v2_glog_cfg_rc})\nCONFIGURE OUTPUT:\n${_pd_v2_glog_cfg_out}\nCONFIGURE ERROR:\n${_pd_v2_glog_cfg_err}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${_pd_v2_glog_bld} --target install
    RESULT_VARIABLE _pd_v2_glog_bld_rc
    OUTPUT_VARIABLE _pd_v2_glog_bld_out
    ERROR_VARIABLE _pd_v2_glog_bld_err)
  if(NOT _pd_v2_glog_bld_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 glog build/install failed (exit ${_pd_v2_glog_bld_rc})\nBUILD OUTPUT:\n${_pd_v2_glog_bld_out}\nBUILD ERROR:\n${_pd_v2_glog_bld_err}")
  endif()
endif()

# jsoncpp. Mooncake's FindJsonCpp.cmake first tries config mode and expects
# JsonCpp::JsonCpp. Official jsoncpp 1.9.5 installs jsoncppConfig.cmake plus
# jsoncpp-namespaced-targets.cmake, which provides that target.
set(_PD_V2_JSONCPP_REF "1.9.5")
set(_PD_V2_JSONCPP_SRC ${THIRD_PARTY_PATH}/install/jsoncpp_pd_v2)
if(NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib/cmake/jsoncpp/jsoncppConfig.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib64/cmake/jsoncpp/jsoncppConfig.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/share/cmake/jsoncpp/jsoncppConfig.cmake")
  if(NOT EXISTS "${_PD_V2_JSONCPP_SRC}/CMakeLists.txt")
    message(STATUS "Fetching jsoncpp ${_PD_V2_JSONCPP_REF} for Mooncake tent sub-build")
    FetchContent_Populate(download_jsoncpp_pd_v2
        GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
        GIT_TAG ${_PD_V2_JSONCPP_REF}
        GIT_SHALLOW TRUE
        UPDATE_DISCONNECTED TRUE
        SOURCE_DIR ${_PD_V2_JSONCPP_SRC})
  endif()
  message(STATUS "Installing jsoncpp ${_PD_V2_JSONCPP_REF} for Mooncake tent sub-build at ${_PD_V2_DEPS_PREFIX}")
  set(_pd_v2_jsoncpp_bld ${PD_V2_MOONCAKE_BUILD_DIR}/jsoncpp-build)
  file(MAKE_DIRECTORY ${_pd_v2_jsoncpp_bld})
  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_STATIC_LIBS=ON
            -DBUILD_OBJECT_LIBS=OFF
            -DJSONCPP_WITH_TESTS=OFF
            -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF
            -DJSONCPP_WITH_EXAMPLE=OFF
            -DJSONCPP_WITH_WARNING_AS_ERROR=OFF
            -DJSONCPP_WITH_STRICT_ISO=OFF
            -DJSONCPP_WITH_CMAKE_PACKAGE=ON
            -DJSONCPP_WITH_PKGCONFIG_SUPPORT=OFF
            -DCMAKE_INSTALL_PREFIX=${_PD_V2_DEPS_PREFIX}
            -DCMAKE_INSTALL_LIBDIR=lib
            -S ${_PD_V2_JSONCPP_SRC}
            -B ${_pd_v2_jsoncpp_bld}
    RESULT_VARIABLE _pd_v2_jsoncpp_cfg_rc
    OUTPUT_VARIABLE _pd_v2_jsoncpp_cfg_out
    ERROR_VARIABLE _pd_v2_jsoncpp_cfg_err)
  if(NOT _pd_v2_jsoncpp_cfg_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 jsoncpp configure failed (exit ${_pd_v2_jsoncpp_cfg_rc})\nCONFIGURE OUTPUT:\n${_pd_v2_jsoncpp_cfg_out}\nCONFIGURE ERROR:\n${_pd_v2_jsoncpp_cfg_err}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${_pd_v2_jsoncpp_bld} --target install
    RESULT_VARIABLE _pd_v2_jsoncpp_bld_rc
    OUTPUT_VARIABLE _pd_v2_jsoncpp_bld_out
    ERROR_VARIABLE _pd_v2_jsoncpp_bld_err)
  if(NOT _pd_v2_jsoncpp_bld_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 jsoncpp build/install failed (exit ${_pd_v2_jsoncpp_bld_rc})\nBUILD OUTPUT:\n${_pd_v2_jsoncpp_bld_out}\nBUILD ERROR:\n${_pd_v2_jsoncpp_bld_err}")
  endif()
  if(EXISTS "${_PD_V2_DEPS_PREFIX}/lib/cmake/jsoncpp/jsoncppConfig.cmake"
     AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib/cmake/jsoncpp/JsonCppConfig.cmake")
    file(WRITE "${_PD_V2_DEPS_PREFIX}/lib/cmake/jsoncpp/JsonCppConfig.cmake"
         "include(\"\${CMAKE_CURRENT_LIST_DIR}/jsoncppConfig.cmake\")\n")
  endif()
endif()
if(EXISTS "${_PD_V2_DEPS_PREFIX}/lib/cmake/jsoncpp/jsoncppConfig.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib/cmake/jsoncpp/JsonCppConfig.cmake")
  file(WRITE "${_PD_V2_DEPS_PREFIX}/lib/cmake/jsoncpp/JsonCppConfig.cmake"
       "include(\"\${CMAKE_CURRENT_LIST_DIR}/jsoncppConfig.cmake\")\n")
endif()

# hiredis. Mooncake TENT's Redis metastore does `find_library(hiredis)`, and
# headers include <hiredis/hiredis.h>. Install hiredis into the same private
# prefix so both the sub-build and final connector link are independent of
# host-provided hiredis-devel.
set(_PD_V2_HIREDIS_REF "v1.2.0")
set(_PD_V2_HIREDIS_SRC ${THIRD_PARTY_PATH}/install/hiredis_pd_v2)
if(NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib/libhiredis.a")
  if(NOT EXISTS "${_PD_V2_HIREDIS_SRC}/CMakeLists.txt")
    message(STATUS "Fetching hiredis ${_PD_V2_HIREDIS_REF} for Mooncake tent sub-build")
    FetchContent_Populate(download_hiredis_pd_v2
        GIT_REPOSITORY https://github.com/redis/hiredis.git
        GIT_TAG ${_PD_V2_HIREDIS_REF}
        GIT_SHALLOW TRUE
        UPDATE_DISCONNECTED TRUE
        SOURCE_DIR ${_PD_V2_HIREDIS_SRC})
  endif()
  message(STATUS "Installing hiredis ${_PD_V2_HIREDIS_REF} for Mooncake tent sub-build at ${_PD_V2_DEPS_PREFIX}")
  set(_pd_v2_hiredis_bld ${PD_V2_MOONCAKE_BUILD_DIR}/hiredis-build)
  file(MAKE_DIRECTORY ${_pd_v2_hiredis_bld})
  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DDISABLE_TESTS=ON
            -DENABLE_EXAMPLES=OFF
            -DENABLE_SSL=OFF
            -DCMAKE_INSTALL_PREFIX=${_PD_V2_DEPS_PREFIX}
            -DCMAKE_INSTALL_LIBDIR=lib
            -S ${_PD_V2_HIREDIS_SRC}
            -B ${_pd_v2_hiredis_bld}
    RESULT_VARIABLE _pd_v2_hiredis_cfg_rc
    OUTPUT_VARIABLE _pd_v2_hiredis_cfg_out
    ERROR_VARIABLE _pd_v2_hiredis_cfg_err)
  if(NOT _pd_v2_hiredis_cfg_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 hiredis configure failed (exit ${_pd_v2_hiredis_cfg_rc})\nCONFIGURE OUTPUT:\n${_pd_v2_hiredis_cfg_out}\nCONFIGURE ERROR:\n${_pd_v2_hiredis_cfg_err}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${_pd_v2_hiredis_bld} --target install
    RESULT_VARIABLE _pd_v2_hiredis_bld_rc
    OUTPUT_VARIABLE _pd_v2_hiredis_bld_out
    ERROR_VARIABLE _pd_v2_hiredis_bld_err)
  if(NOT _pd_v2_hiredis_bld_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 hiredis build/install failed (exit ${_pd_v2_hiredis_bld_rc})\nBUILD OUTPUT:\n${_pd_v2_hiredis_bld_out}\nBUILD ERROR:\n${_pd_v2_hiredis_bld_err}")
  endif()
endif()

# yalantinglibs (ylt, header-only). Unlike yaml-cpp / gflags it is NOT
# vendored elsewhere in KsanaLLM, and Mooncake never builds its
# extern/yalantinglibs submodule — its CMake always does
# `find_package(yalantinglibs CONFIG REQUIRED)`, expecting a host install
# (normally provided by Mooncake's dependencies.sh). Fetch the exact
# version Mooncake pins (extern/yalantinglibs @ v0.5.7) and install it
# (just headers + config, no examples/tests) into the shared prefix.
set(_PD_V2_YLT_REF "v0.5.7")
set(_PD_V2_YLT_SRC ${THIRD_PARTY_PATH}/install/yalantinglibs_pd_v2)
if(NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib/cmake/yalantinglibs/yalantinglibsConfig.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/lib64/cmake/yalantinglibs/yalantinglibsConfig.cmake"
   AND NOT EXISTS "${_PD_V2_DEPS_PREFIX}/share/cmake/yalantinglibs/yalantinglibsConfig.cmake")
  if(NOT EXISTS "${_PD_V2_YLT_SRC}/CMakeLists.txt")
    message(STATUS "Fetching yalantinglibs ${_PD_V2_YLT_REF} for Mooncake tent sub-build")
    FetchContent_Populate(download_yalantinglibs_pd_v2
        GIT_REPOSITORY https://github.com/alibaba/yalantinglibs.git
        GIT_TAG ${_PD_V2_YLT_REF}
        GIT_SHALLOW TRUE
        UPDATE_DISCONNECTED TRUE
        SOURCE_DIR ${_PD_V2_YLT_SRC})
  endif()
  message(STATUS "Installing yalantinglibs ${_PD_V2_YLT_REF} for Mooncake tent sub-build at ${_PD_V2_DEPS_PREFIX}")
  set(_pd_v2_ylt_bld ${PD_V2_MOONCAKE_BUILD_DIR}/yalantinglibs-build)
  file(MAKE_DIRECTORY ${_pd_v2_ylt_bld})
  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_EXAMPLES=OFF
            -DBUILD_BENCHMARK=OFF
            -DBUILD_UNIT_TESTS=OFF
            -DCMAKE_INSTALL_PREFIX=${_PD_V2_DEPS_PREFIX}
            -DCMAKE_INSTALL_LIBDIR=lib
            -S ${_PD_V2_YLT_SRC}
            -B ${_pd_v2_ylt_bld}
    RESULT_VARIABLE _pd_v2_ylt_cfg_rc
    OUTPUT_VARIABLE _pd_v2_ylt_cfg_out
    ERROR_VARIABLE _pd_v2_ylt_cfg_err)
  if(NOT _pd_v2_ylt_cfg_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 yalantinglibs configure failed (exit ${_pd_v2_ylt_cfg_rc})\nCONFIGURE OUTPUT:\n${_pd_v2_ylt_cfg_out}\nCONFIGURE ERROR:\n${_pd_v2_ylt_cfg_err}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${_pd_v2_ylt_bld} --target install
    RESULT_VARIABLE _pd_v2_ylt_bld_rc
    OUTPUT_VARIABLE _pd_v2_ylt_bld_out
    ERROR_VARIABLE _pd_v2_ylt_bld_err)
  if(NOT _pd_v2_ylt_bld_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 yalantinglibs build/install failed (exit ${_pd_v2_ylt_bld_rc})\nBUILD OUTPUT:\n${_pd_v2_ylt_bld_out}\nBUILD ERROR:\n${_pd_v2_ylt_bld_err}")
  endif()
endif()

# Prepend the vendored prefix to whatever CMAKE_PREFIX_PATH the parent
# build already set, so the tent sub-build finds our deps first while
# still inheriting host-provided dependency locations. Escape the list
# separator so the multi-entry path reaches the child cmake as a single
# -D argument instead of being split into separate execute_process args.
set(_PD_V2_MOONCAKE_PREFIX_PATH "${_PD_V2_DEPS_PREFIX}")
if(CMAKE_PREFIX_PATH)
  list(APPEND _PD_V2_MOONCAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
endif()
string(REPLACE ";" "\\;" _PD_V2_MOONCAKE_PREFIX_PATH_ARG "${_PD_V2_MOONCAKE_PREFIX_PATH}")

# Stamp file so we don't re-configure Mooncake on every cmake run. Bump
# the suffix below to force a clean rebuild (e.g. after a GIT_TAG change).
set(_pd_v2_mc_stamp "${PD_V2_MOONCAKE_BUILD_DIR}/.pd_v2_stamp_v1")

if(NOT EXISTS "${_pd_v2_mc_stamp}"
   OR NOT EXISTS "${PD_V2_MOONCAKE_BUILD_DIR}/mooncake-transfer-engine/src/libtransfer_engine.a")

  message(STATUS "Building vendored Mooncake transfer-engine (tent) at ${PD_V2_MOONCAKE_BUILD_DIR}")
  file(MAKE_DIRECTORY ${PD_V2_MOONCAKE_BUILD_DIR})

  # GCC 12 -Werror=maybe-uninitialized false-positive on ylt coro_rpc's
  # `async_rpc_raw_result{ec}` temporary (libstdc++ basic_string SSO
  # heuristic) — promoted to error by Mooncake's own warning flags on
  # some builds. Demote to warning so the vendored sub-build is portable
  # across GCC 12 minor versions / ylt minor versions without forcing
  # users to patch the upstream sources.
  #
  # glog/logging.h does `#include <gflags/gflags.h>`. Mooncake's `base`
  # target (status.cpp) uses glog but does NOT link the gflags target, so
  # it never inherits gflags' include dir. When gflags resolves to a
  # non-default prefix (a system find_package hit under conda/etc., or our
  # private deps prefix), the bare `-I.../include` isn't enough. Resolve
  # gflags' real include dir and add it explicitly to the sub-build flags.
  #   - prefix has gflags.h -> already covered by -I${_PD_V2_DEPS_PREFIX}/include
  #   - system gflags       -> find_package / find_path locates the headers
  #   - no gflags at all     -> WARNING (user must install libgflags-dev)
  set(_PD_V2_MC_EXTRA_INC "")
  if(NOT EXISTS "${_PD_V2_DEPS_PREFIX}/include/gflags/gflags.h")
    # NO_CMAKE_PACKAGE_REGISTRY: same rationale as external/gflags.cmake —
    # never resolve via the polluting ~/.cmake/packages/gflags/ registry.
    find_package(gflags QUIET NO_CMAKE_PACKAGE_REGISTRY)
    if(GFLAGS_INCLUDE_DIR AND EXISTS "${GFLAGS_INCLUDE_DIR}/gflags/gflags.h")
      set(_PD_V2_MC_EXTRA_INC " -I${GFLAGS_INCLUDE_DIR}")
    else()
      find_path(_PD_V2_GFLAGS_INC NAMES gflags/gflags.h)
      if(_PD_V2_GFLAGS_INC)
        set(_PD_V2_MC_EXTRA_INC " -I${_PD_V2_GFLAGS_INC}")
      endif()
    endif()
    if(_PD_V2_MC_EXTRA_INC)
      message(STATUS "pd_v2 Mooncake: adding gflags include dir${_PD_V2_MC_EXTRA_INC}")
    else()
      message(WARNING
        "pd_v2 Mooncake: could not locate gflags/gflags.h anywhere. If the "
        "build fails with 'gflags/gflags.h: No such file or directory', "
        "install libgflags-dev (Debian/Ubuntu) or gflags-devel (RHEL/CentOS).")
    endif()
  endif()
  set(_PD_V2_MC_CXX_FLAGS
      "-I${_PD_V2_DEPS_PREFIX}/include${_PD_V2_MC_EXTRA_INC} -Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")

  # Probe rdma-core capability instead of forcing RDMA off everywhere. Mooncake's
  # RDMA transport uses APIs that only exist on a recent libibverbs: the
  # IBV_ACCESS_RELAXED_ORDERING access flag (rdma-core >= 25) and
  # ibv_query_gid_ex() / struct ibv_gid_entry (rdma-core >= 28). On hosts whose
  # rdma-core is too old (e.g. CentOS 7's libibverbs) those are missing and the
  # RDMA path won't compile, so we fall back to the TCP tent transport there;
  # on hosts with a modern rdma-core we keep full RDMA. Honor an explicit
  # -DUSE_RDMA=... from the caller if provided.
  if(DEFINED USE_RDMA)
    set(_PD_V2_MC_USE_RDMA ${USE_RDMA})
    message(STATUS "pd_v2 Mooncake: USE_RDMA=${_PD_V2_MC_USE_RDMA} (explicitly set by caller)")
  else()
    include(CheckCXXSourceCompiles)
    set(_pd_v2_save_req_quiet "${CMAKE_REQUIRED_QUIET}")
    set(_pd_v2_save_req_libs "${CMAKE_REQUIRED_LIBRARIES}")
    set(CMAKE_REQUIRED_QUIET ON)
    set(CMAKE_REQUIRED_LIBRARIES ibverbs)
    check_cxx_source_compiles(
      "#include <infiniband/verbs.h>\nint main() {\n  struct ibv_gid_entry e;\n  (void)e;\n  unsigned a = IBV_ACCESS_RELAXED_ORDERING;\n  (void)a;\n  (void)&ibv_query_gid_ex;\n  return 0;\n}"
      _KLLM_PD_V2_RDMA_OK)
    set(CMAKE_REQUIRED_QUIET "${_pd_v2_save_req_quiet}")
    set(CMAKE_REQUIRED_LIBRARIES "${_pd_v2_save_req_libs}")
    if(_KLLM_PD_V2_RDMA_OK)
      set(_PD_V2_MC_USE_RDMA ON)
      message(STATUS "pd_v2 Mooncake: rdma-core is new enough -> USE_RDMA=ON")
    else()
      set(_PD_V2_MC_USE_RDMA OFF)
      message(STATUS
        "pd_v2 Mooncake: rdma-core too old or absent (missing ibv_query_gid_ex / "
        "IBV_ACCESS_RELAXED_ORDERING) -> USE_RDMA=OFF, falling back to TCP tent transport")
    endif()
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DCMAKE_BUILD_TYPE=Release
            -DUSE_TENT=ON
            -DUSE_CUDA=${_PD_V2_MOONCAKE_USE_CUDA}
            -DUSE_RDMA=${_PD_V2_MC_USE_RDMA}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
            -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
            -DCMAKE_PREFIX_PATH=${_PD_V2_MOONCAKE_PREFIX_PATH_ARG}
            -DBUILD_UNIT_TESTS=OFF
            -DBUILD_EXAMPLES=OFF
            -DBUILD_BENCHMARK=OFF
            -DBUILD_TENT_TESTS=OFF
            "-DCMAKE_CXX_FLAGS=${_PD_V2_MC_CXX_FLAGS}"
            -DCMAKE_SHARED_LINKER_FLAGS=-L${_PD_V2_DEPS_PREFIX}/lib
            -DCMAKE_MODULE_LINKER_FLAGS=-L${_PD_V2_DEPS_PREFIX}/lib
            -DCMAKE_EXE_LINKER_FLAGS=-L${_PD_V2_DEPS_PREFIX}/lib
            -S ${PD_V2_MOONCAKE_SOURCE_DIR}/mooncake-transfer-engine
            -B ${PD_V2_MOONCAKE_BUILD_DIR}
    RESULT_VARIABLE _pd_v2_mc_cfg_rc
    OUTPUT_VARIABLE _pd_v2_mc_cfg_out
    ERROR_VARIABLE _pd_v2_mc_cfg_err)
  if(NOT _pd_v2_mc_cfg_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 Mooncake transfer-engine configure failed (exit ${_pd_v2_mc_cfg_rc})\nCONFIGURE OUTPUT:\n${_pd_v2_mc_cfg_out}\nCONFIGURE ERROR:\n${_pd_v2_mc_cfg_err}")
  endif()

  include(ProcessorCount)
  ProcessorCount(_pd_v2_nproc)
  if(_pd_v2_nproc EQUAL 0)
    set(_pd_v2_nproc 8)
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${PD_V2_MOONCAKE_BUILD_DIR} --parallel ${_pd_v2_nproc}
    RESULT_VARIABLE _pd_v2_mc_bld_rc
    OUTPUT_VARIABLE _pd_v2_mc_bld_out
    ERROR_VARIABLE _pd_v2_mc_bld_err)
  if(NOT _pd_v2_mc_bld_rc EQUAL 0)
    message(FATAL_ERROR "pd_v2 Mooncake transfer-engine build failed (exit ${_pd_v2_mc_bld_rc})\nBUILD OUTPUT:\n${_pd_v2_mc_bld_out}\nBUILD ERROR:\n${_pd_v2_mc_bld_err}")
  endif()

  file(WRITE "${_pd_v2_mc_stamp}" "ok")
endif()

# Expose include + lib roots to pd_v2/mooncake/CMakeLists.txt.
set(PD_V2_MOONCAKE_DEPS_PREFIX ${_PD_V2_DEPS_PREFIX})
set(PD_V2_MOONCAKE_INCLUDE_DIRS
    ${PD_V2_MOONCAKE_SOURCE_DIR}/mooncake-transfer-engine/include
    ${PD_V2_MOONCAKE_SOURCE_DIR}/mooncake-transfer-engine/tent/include
    ${PD_V2_MOONCAKE_DEPS_PREFIX}/include)

message(STATUS "pd_v2 Mooncake source: ${PD_V2_MOONCAKE_SOURCE_DIR}")
message(STATUS "pd_v2 Mooncake build:  ${PD_V2_MOONCAKE_BUILD_DIR}")
