/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/service/service_lifetime_interface.h"

namespace ksana_llm {

// A dummy lifetime implementation that do nothing.
// Used for unit tests, do nothing if not set.
class DummyServiceLifetimeManager : public ServiceLifetimeManagerInterface {
 public:
  ~DummyServiceLifetimeManager() {}
  virtual void ShutdownService() {}
};

void SetServiceLifetimeManager(std::shared_ptr<ServiceLifetimeManagerInterface> lifetime_manager);

// Used to stop current inference server.
std::shared_ptr<ServiceLifetimeManagerInterface> GetServiceLifetimeManager();

}  // namespace ksana_llm
