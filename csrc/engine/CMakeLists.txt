# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE engine_SRCS
	${PROJECT_SOURCE_DIR}/csrc/engine/*.cpp)
list(FILTER engine_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "engine_SRCS: ${engine_SRCS}")

add_library(engine STATIC ${engine_SRCS})
# pd_v2 Decode hook + Prefill intake are referenced by name in
# inference_engine.cpp (dlopen of libpd_v2_connector_<backend>.so for the
# transport-only ABI; the hook + intake business logic live in
# pd_v2_decode / pd_v2_prefill which are linked statically). libdl
# provides dlopen/dlsym for the runtime backend load.
target_link_libraries(engine PUBLIC pd_v2_decode pd_v2_prefill pd_v2_public endpoints ${CMAKE_DL_LIBS})

# for test
file(GLOB_RECURSE engine_test_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/engine/*test.cpp)

message(STATUS "runtime_test_SRCS: ${runtime_test_SRCS}")
cpp_test(engine_test SRCS ${engine_test_SRCS} DEPS engine executor)
