# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# Header-only: PdV2RuntimeConfig (inference_engine / inference_executor
# process-internal runtime config) + pd_v2_connector_config_builder (maps
# PdV2RuntimeConfig -> the pd_v2_public ABI PdV2ConnectorConfig). The builder
# header includes pd_v2/public/pd_v2_config.h, hence the pd_v2_public link.
add_library(pd_v2_configure INTERFACE)
target_include_directories(pd_v2_configure INTERFACE
    ${PROJECT_SOURCE_DIR}/csrc
)
target_link_libraries(pd_v2_configure INTERFACE utils pd_v2_public)
