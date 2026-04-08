/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "global_cache_connector/global_cache_connector_factory.h"

#include "global_cache_connector/mooncake_connector.h"
#include "global_cache_connector/mooncake_mock_connector.h"
#include "utils/logger.h"

namespace ksana_llm {

std::shared_ptr<GlobalCacheConnectorInterface> GlobalCacheConnectorFactory::CreateConnector(
    const GlobalCacheConnectorConfig& config) {
  GlobalCacheConnectorInterface* raw_ptr = nullptr;

  if (config.backend_type == "mooncake-mock") {
    raw_ptr = new MooncakeMockConnector();
  } else if (config.backend_type == "mooncake") {
    raw_ptr = new MooncakeConnector();
  } else {
    KLLM_LOG_ERROR << "Unknown global cache connector backend: " << config.backend_type
                   << ". Supported backends: 'mooncake-mock', 'mooncake'";
    return nullptr;
  }

  auto instance = std::shared_ptr<GlobalCacheConnectorInterface>(raw_ptr, [](GlobalCacheConnectorInterface* ptr) {
    if (ptr) {
      ptr->Shutdown();
      delete ptr;
    }
  });

  Status init_status = instance->Initialize(config);
  if (!init_status.OK()) {
    KLLM_LOG_ERROR << "Failed to initialize global cache connector: " << init_status.GetMessage();
    return nullptr;
  }

  KLLM_LOG_INFO << "Global cache connector created and initialized. Config: " << config.ToString();
  return instance;
}

}  // namespace ksana_llm
