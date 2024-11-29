# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

option(WITH_TRPC_ENDPOINT "Build with trpc endpoint" OFF)

if(WITH_TRPC_ENDPOINT)
    add_subdirectory(trpc)
endif()
