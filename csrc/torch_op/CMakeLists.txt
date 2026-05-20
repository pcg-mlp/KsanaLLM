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

if(WITH_CUDA OR WITH_TOPS)
  # Use --whole-archive to ensure ALL symbols from static libs are exported into the .so,
  # so that test binaries can dynamically link against torch_serving and find all symbols.
  # ksana_llm_libs is INTERFACE so we extract its member libraries via generator expression.
  get_target_property(_ksana_libs ksana_llm_libs INTERFACE_LINK_LIBRARIES)
  target_link_libraries(torch_serving PRIVATE ${TORCH_LIBRARIES}
    -Wl,--whole-archive
    ${_ksana_libs}
    -Wl,--no-whole-archive)
endif()

if(WITH_ACL)
  target_link_libraries(torch_serving PRIVATE ${TORCH_LIBRARIES} ksana_llm_libs -Wl,--whole-archive layers -Wl,--no-whole-archive kernels)
endif()

# for test
if(WITH_STANDALONE_TEST)
  file(GLOB_RECURSE serving_op_test_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/torch_op/*test.cpp)
  message(STATUS "serving_op_test_SRCS: ${serving_op_test_SRCS}")
  cpp_test(serving_op_test SRCS ${serving_op_test_SRCS} DEPS torch_serving runtime utils)
endif()
