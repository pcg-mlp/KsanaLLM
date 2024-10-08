# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE atb_attention_SRCS ./*.cc)

message(STATUS "atb_attention_SRCS: ${atb_attention_SRCS}")

add_library(atb_attention STATIC ${atb_attention_SRCS})
target_link_libraries(atb_attention PRIVATE ${ACL_SHARED_LIBS} ${ACL_SHARED_LIBS})