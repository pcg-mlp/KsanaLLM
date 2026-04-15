/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "global_cache_connector/global_cache_connector_factory.h"

#include <dlfcn.h>

#include "global_cache_connector/mooncake_mock_connector.h"
#include "utils/logger.h"

namespace ksana_llm {

std::shared_ptr<GlobalCacheConnectorInterface> GlobalCacheConnectorFactory::CreateConnector(
    const GlobalCacheConnectorConfig& config) {
  GlobalCacheConnectorInterface* raw_ptr = nullptr;

  if (config.backend_type == "mooncake-mock") {
    raw_ptr = new MooncakeMockConnector();
  } else if (config.backend_type == "mooncake") {
    // Load MooncakeConnector plugin via dlopen (built separately with -DUSE_MOONCAKE_REAL_STORE=ON).
    void* handle = dlopen("libmooncake_connector.so", RTLD_NOW);
    if (!handle) {
      KLLM_LOG_ERROR << "Mooncake backend requested but libmooncake_connector.so not found. "
                     << "Please rebuild with -DUSE_MOONCAKE_REAL_STORE=ON. "
                     << "dlerror: " << dlerror();
      return nullptr;
    }
    using CreateFn = GlobalCacheConnectorInterface* (*)();
    auto create_fn = reinterpret_cast<CreateFn>(dlsym(handle, "CreateMooncakeConnector"));
    if (!create_fn) {
      KLLM_LOG_ERROR << "Failed to find CreateMooncakeConnector symbol in libmooncake_connector.so: " << dlerror();
      dlclose(handle);
      return nullptr;
    }
    raw_ptr = create_fn();
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
