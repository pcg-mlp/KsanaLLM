# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE atb_plugin_operations_SRCS ./*.cc)

message(STATUS "atb_plugin_operations_SRCS: ${atb_plugin_operations_SRCS}")

add_library(atb_plugin_operations STATIC ${atb_plugin_operations_SRCS})
target_link_libraries(atb_plugin_operations PRIVATE ${ACL_SHARED_LIBS} ${ACL_SHARED_LIBS})