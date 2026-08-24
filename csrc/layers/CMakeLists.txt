# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE layers_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/layers/cpu/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/layers/*.cpp)
list(FILTER layers_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/nvidia/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/ascend/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/zixiao/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/kunlun/.*.cpp")
if(NOT WITH_CUDA)
  list(FILTER layers_SRCS EXCLUDE REGEX ".*/flashinfer_.*\\.cpp")
endif()

message(STATUS "layers_SRCS: ${layers_SRCS}")

set(layers_nvidia_SRCS "")
set(layers_nvidia_LIBS "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_nvidia_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/layers/nvidia/*.cpp
    ${PROJECT_SOURCE_DIR}/csrc/layers/nvidia/*.cu)
  list(FILTER layers_nvidia_SRCS EXCLUDE REGEX ".*test\.cpp")
  list(FILTER layers_nvidia_SRCS EXCLUDE REGEX ".*test\.cu")
  list(APPEND layers_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)
endif()

set(layers_ascend_SRCS "")
set(layers_ascend_LIBS "")

if(WITH_ACL)
  file(GLOB_RECURSE layers_ascend_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/ascend/*.cpp)
  list(APPEND layers_ascend_LIBS ${ACL_SHARED_LIBS} llm_kernels_ascend_utils_common
    atb_attention)
endif()

set(layers_zixiao_SRCS "")
set(layers_zixiao_LIBS "")

if(WITH_TOPS)
  file(GLOB_RECURSE layers_zixiao_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/zixiao/*.cpp)
  list(APPEND layers_zixiao_LIBS ${TOPS_SHARED_LIBS})
endif()

# NOTE(fangtang, kunlun-adapt): Kunlun XPU layers.
set(layers_kunlun_SRCS "")
set(layers_kunlun_LIBS "")

if(WITH_KUNLUN)
  file(GLOB_RECURSE layers_kunlun_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/kunlun/*.cpp)
  list(FILTER layers_kunlun_SRCS EXCLUDE REGEX ".*test.cpp")
  list(APPEND layers_kunlun_LIBS ${XPU_SHARED_LIBS})
endif()

add_library(layers STATIC ${layers_SRCS} ${layers_nvidia_SRCS} ${layers_ascend_SRCS}
                          ${layers_zixiao_SRCS} ${layers_kunlun_SRCS})

target_link_libraries(layers PUBLIC
  kernels ${layers_nvidia_LIBS} ${layers_ascend_LIBS} ${layers_kunlun_LIBS}
  layer_progress_tracker pd_v2_layer_event_tracker runtime
)

# for test
set(layers_test_nvidia_SRCS "")
set(layers_test_ascend_SRCS "")
set(layers_test_zixiao_SRCS "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_test_nvidia_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/nvidia/*test.cpp)
endif()

if(WITH_ACL)
  file(GLOB_RECURSE layers_test_ascend_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/ascend/*test.cpp)
endif()

set(layers_test_common_DEPS layers runtime utils executor)

cpp_test(matmul_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/matmul_layer_test.cpp
  DEPS ${layers_test_common_DEPS})

cpp_test(moe_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/moe_layer_test.cpp
       ${layers_test_nvidia_SRCS}
  DEPS ${layers_test_common_DEPS})

cpp_test(attention_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/attention_layer_test.cpp
  DEPS ${layers_test_common_DEPS})

cpp_test(misc_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/misc_layer_test.cpp
       ${layers_test_ascend_SRCS}
  DEPS ${layers_test_common_DEPS})

cpp_test(wfp4_moe_layers_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/tests/wfp4_moe_layer_test.cpp
  DEPS ${layers_test_common_DEPS})

cpp_test(humming_search_cache_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/tests/humming_search_cache_test.cpp
  DEPS ${layers_test_common_DEPS})

# topk_layers_test and grouped_topk_layer_test are CUDA-only: they include nvtx / cuda_runtime.h
# and use cudaStream APIs + CUDA-only GroupedTopkLayer, so they only build under WITH_CUDA.
if(WITH_CUDA)
  cpp_test(topk_layers_test
    SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/nvidia/topk_layer_test.cpp
    DEPS ${layers_test_common_DEPS})

  cpp_test(grouped_topk_layer_test
    SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/tests/grouped_topk_layer_test.cpp
    DEPS ${layers_test_common_DEPS})
endif()

# Kunlun XPU layer test — full framework link
if(WITH_KUNLUN AND WITH_TESTING)
  # Get torch lib path (used by both test targets).
  # Use find_spec instead of import torch to avoid stdout pollution from
  # torch_xmlir import hooks ("WARNING: Libraries loaded from different
  # directories!").  Corrupted _TORCH_LIB_DIR breaks -L flags and rpaths.
  execute_process(
    COMMAND ${_PYBIN} -c "import importlib.util;spec=importlib.util.find_spec('torch');print(spec.submodule_search_locations[0]+'/lib',end='')"
    OUTPUT_VARIABLE _TORCH_LIB_DIR ERROR_QUIET)

  # Separate bkcl_comm_test (standalone, no gtest dependency)
  if(EXISTS ${PROJECT_SOURCE_DIR}/csrc/layers/kunlun/kunlun_bkcl_comm_test.cpp)
    add_executable(bkcl_comm_test
      ${PROJECT_SOURCE_DIR}/csrc/layers/kunlun/kunlun_bkcl_comm_test.cpp)
    target_include_directories(bkcl_comm_test PRIVATE
      ${PROJECT_SOURCE_DIR}/csrc
      ${PROJECT_SOURCE_DIR}/tests)
    target_link_libraries(bkcl_comm_test
      -Wl,--start-group
      layers kernels device utils configure cache_manager
      loguru fmt
      -Wl,--end-group
      -pthread
      -L${_TORCH_LIB_DIR} -ltorch -ltorch_cpu -lc10
      -L${XMLIR_DIR} -L${XMLIR_DIR}/xre/so
      -lxpu_dnn -lxpuapi -lbkcl -lxpurt -lcudart)
    target_link_directories(bkcl_comm_test PRIVATE ${XPU_LIB_DIRS} ${_TORCH_LIB_DIR})
    set_target_properties(bkcl_comm_test PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
      BUILD_RPATH "${_TORCH_LIB_DIR};${XMLIR_DIR};${XMLIR_DIR}/xre/so")
  endif()

  # Other kunlun layer tests (exclude bkcl_comm_test)
  file(GLOB kunlun_layers_test_SRCS ${PROJECT_SOURCE_DIR}/csrc/layers/kunlun/*test.cpp)
  list(FILTER kunlun_layers_test_SRCS EXCLUDE REGEX ".*bkcl_comm_test.cpp")
  if(kunlun_layers_test_SRCS)
    add_executable(kunlun_layer_test ${kunlun_layers_test_SRCS})
    set(_KUNLUN_RUNTIME_RPATH
      "${_TORCH_LIB_DIR};${XMLIR_DIR};${XMLIR_DIR}/xre/so;${XMLIR_DIR}/xhpc/xblas/dependency_so")
    # libbkcl.so lives in ${XMLIR_DIR}, but libxpuml/libxpucuda are in xre/so
    # and sometimes xhpc/xblas/dependency_so; make both direct search entries.
    target_link_options(kunlun_layer_test PRIVATE
      -Wl,--disable-new-dtags
      -Wl,-rpath,${XMLIR_DIR}/xre/so
      -Wl,-rpath,${XMLIR_DIR}/xhpc/xblas/dependency_so)
    target_include_directories(kunlun_layer_test PRIVATE
      ${PROJECT_SOURCE_DIR}/tests
      ${gtest_SOURCE_DIR}/include
      ${gmock_SOURCE_DIR}/include)
    # Link order matters: static archives first, then shared libs at the end
    # so the linker can resolve symbols from .a in the .so's that follow.
    # Link without --whole-archive: linker only pulls in .o files that are actually referenced
    target_link_libraries(kunlun_layer_test
      -Wl,--start-group
      layers kernels device utils cache_manager configure
      loguru fmt
      gtest_main gtest gmock
      -Wl,--end-group
      -pthread
      -L${_TORCH_LIB_DIR} -ltorch -ltorch_cpu -lc10
      -L${XMLIR_DIR} -L${XMLIR_DIR}/xre/so
      -lxpu_dnn -lxpuapi -lbkcl -lxpurt -lcudart
      -Wl,--no-as-needed -lxpuml -lxpucuda -Wl,--as-needed)
    target_link_directories(kunlun_layer_test PRIVATE ${XPU_LIB_DIRS} ${_TORCH_LIB_DIR})
    set_target_properties(kunlun_layer_test PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
      BUILD_RPATH "${_KUNLUN_RUNTIME_RPATH}")
  endif()
endif()
