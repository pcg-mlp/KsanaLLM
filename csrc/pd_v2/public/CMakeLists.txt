# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# Header-only ABI: PdV2ConnectorInterface / PdV2ConnectorConfig / pd_v2_types.
add_library(pd_v2_public INTERFACE)
target_include_directories(pd_v2_public INTERFACE
    ${PROJECT_SOURCE_DIR}/csrc
)
target_link_libraries(pd_v2_public INTERFACE utils)
