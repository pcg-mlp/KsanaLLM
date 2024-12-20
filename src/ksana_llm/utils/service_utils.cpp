/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/service_utils.h"

namespace ksana_llm {

std::shared_ptr<ServiceLifetimeManagerInterface> g_lifetime_manager = nullptr;

void SetServiceLifetimeManager(std::shared_ptr<ServiceLifetimeManagerInterface> lifetime_manager) {
  g_lifetime_manager = lifetime_manager;
}

std::shared_ptr<ServiceLifetimeManagerInterface> GetServiceLifetimeManager() {
  if (g_lifetime_manager == nullptr) {
    g_lifetime_manager = std::make_shared<DummyServiceLifetimeManager>();
  }

  return g_lifetime_manager;
}

}  // namespace ksana_llm
