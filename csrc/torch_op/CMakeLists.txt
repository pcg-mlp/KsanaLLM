# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE torch_op_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/torch_op/*.cpp)
list(FILTER torch_op_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "torch_op_SRCS: ${torch_op_SRCS}")

add_library(torch_serving SHARED ${torch_op_SRCS} ${host_stub_SRCS})

set_target_properties(torch_serving PROPERTIES
  BUILD_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib"
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib"
  INSTALL_RPATH_USE_LINK_PATH OFF
)

# Recursively expand INTERFACE aggregator targets (e.g. `pd_v2`) into their concrete
# sub-libraries and drop embedded -Wl,* linker-flag tokens, so every real archive becomes
# an explicit item we can keep inside a --whole-archive bracket. An INTERFACE target name
# placed between --whole-archive/--no-whole-archive does NOT reliably keep its expanded
# sub-archives inside the bracket: CMake de-duplicates/reorders those targets out of it,
# and their objects never make it into the .so.
function(_kllm_flatten_link_libs _out)
  set(_acc "")
  foreach(_it IN LISTS ARGN)
    if(_it MATCHES "^-Wl,")
      continue()
    endif()
    if(TARGET ${_it})
      get_target_property(_ty ${_it} TYPE)
      if(_ty STREQUAL "INTERFACE_LIBRARY")
        get_target_property(_deps ${_it} INTERFACE_LINK_LIBRARIES)
        if(_deps)
          _kllm_flatten_link_libs(_sub ${_deps})
          list(APPEND _acc ${_sub})
        endif()
      else()
        list(APPEND _acc ${_it})
      endif()
    else()
      list(APPEND _acc ${_it})
    endif()
  endforeach()
  set(${_out} "${_acc}" PARENT_SCOPE)
endfunction()

if(WITH_CUDA OR WITH_TOPS)
  # Use --whole-archive to ensure ALL symbols from static libs are exported into the .so,
  # so that test binaries (which link ONLY torch_serving.so) can resolve every symbol —
  # including those referenced solely by unit tests.
  get_target_property(_ksana_libs ksana_llm_libs INTERFACE_LINK_LIBRARIES)
  # Third-party libs (gflags/loguru/NCCL) MUST stay out of --whole-archive and be linked
  # normally: gflags registers global command-line flags (flagfile, ...) in a static
  # initializer; force-including its objects here while another binary also static-links
  # gflags makes the flag register twice and aborts at startup. Split them out FIRST (by
  # name) so they are never expanded/whole-archived — gflags may itself be an INTERFACE
  # target (system-found case), which the flattening below would otherwise unwrap.
  set(_kllm_plain_libs loguru gflags ${NCCL_LIBRARIES})
  set(_ksana_to_whole "")
  set(_ksana_libs_plain "")
  foreach(_lib IN LISTS _ksana_libs)
    if(_lib MATCHES "^-Wl,")
      continue()
    endif()
    if("${_lib}" IN_LIST _kllm_plain_libs)
      list(APPEND _ksana_libs_plain ${_lib})
    else()
      list(APPEND _ksana_to_whole ${_lib})
    endif()
  endforeach()
  # Flatten the to-whole set: this turns the `pd_v2` INTERFACE aggregator into its concrete
  # static sub-libraries (pd_v2_decode, pd_v2_wire, ...) so they actually land
  # inside the --whole-archive bracket. Only top-level INTERFACE targets are expanded;
  # static libs are kept as-is (so their own deps on gflags etc. are NOT pulled into whole).
  _kllm_flatten_link_libs(_ksana_libs_whole ${_ksana_to_whole})
  list(REMOVE_DUPLICATES _ksana_libs_whole)
  target_link_libraries(torch_serving PRIVATE ${TORCH_LIBRARIES}
    -Wl,--whole-archive
    ${_ksana_libs_whole}
    -Wl,--no-whole-archive
    ${_ksana_libs_plain})
endif()

if(WITH_ACL)
  target_link_libraries(torch_serving PRIVATE ${TORCH_LIBRARIES} ksana_llm_libs -Wl,--whole-archive layers -Wl,--no-whole-archive kernels)
endif()

if(WITH_KUNLUN)
  # torch_serving.so is dlopen'd by the Python interpreter, so Python C API symbols
  # (from libtorch_python.so) are resolved at runtime from the running interpreter.
  # --allow-shlib-undefined suppresses the link-time "undefined reference" errors
  # that come from libtorch_python.so's dependency on libpython.
  target_link_options(torch_serving PRIVATE -Wl,--allow-shlib-undefined)

  # Kunlun: link torch libs + all internal static libs with --whole-archive for device
  # to ensure template specializations (T<3>) are not discarded by the linker.
  # Use --start-group/--end-group to resolve circular dependencies among static libs.
  target_link_libraries(torch_serving PRIVATE
    ${TORCH_LIBRARIES}
    -Wl,--whole-archive layers -Wl,--no-whole-archive
    -Wl,--start-group
    kernels
    service
    supervisor
    engine
    executor
    ipc
    -Wl,--whole-archive batch_scheduler -Wl,--no-whole-archive
    cache_manager
    distributed
    # runtime holds TrieTree / TrieGenerator / MtpDecoderAdapter / DSparkDecoder /
    # MtpStateGuard etc.; whole-archive so those symbols are exported from torch_serving.so
    # for the decoder-adapter / draft_generator unit tests (which link ONLY torch_serving.so).
    -Wl,--whole-archive runtime -Wl,--no-whole-archive
    configure
    endpoints
    samplers
    profiler
    periphery
    model_loader
    models
    modules
    # model_base holds BaseWeight's ctor; production never constructs the base directly
    # (only concrete weight subclasses do), so it isn't pulled in normally. whole-archive
    # so BaseWeight's symbols are exported for module_mla_test's TestWeight subclass.
    -Wl,--whole-archive model_base -Wl,--no-whole-archive
    connector
    global_cache_connector
    transfer
    -Wl,--whole-archive device -Wl,--no-whole-archive
    # utils holds RuntimeDllManager (needed by utils_test); whole-archive to export it.
    -Wl,--whole-archive utils -Wl,--no-whole-archive
    # pd_v2 concrete sub-libraries (PeerSelector / LinUCBRouter / wire codecs). The pd_v2
    # aggregator is an INTERFACE target and cannot sit inside a --whole-archive bracket
    # reliably, so its sub-libs are listed explicitly and whole-archived for the pd_v2 tests.
    -Wl,--whole-archive
    pd_v2_public pd_v2_configure pd_v2_wire pd_v2_decode pd_v2_prefill
    -Wl,--no-whole-archive
    -Wl,--end-group
    loguru
    gflags
    fmt
    yaml-cpp
    re2
    ${Python3_LIBRARIES}
    # libxpuml/libxpucuda are indirect dependencies of libbkcl.so
    # (bkcl NEEDs them, but they reside in xre/so/ while bkcl's RUNPATH is only $ORIGIN).
    # --no-as-needed forces them into DT_NEEDED so the dynamic linker resolves them
    # via libtorch_serving.so's RUNPATH (which includes xre/so).
    -Wl,--no-as-needed -lxpuml -lxpucuda -Wl,--as-needed
  )
endif()

# for test
if(WITH_STANDALONE_TEST)
  file(GLOB_RECURSE serving_op_test_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/torch_op/*test.cpp)
  message(STATUS "serving_op_test_SRCS: ${serving_op_test_SRCS}")
  cpp_test(serving_op_test SRCS ${serving_op_test_SRCS} DEPS torch_serving runtime utils)
endif()
