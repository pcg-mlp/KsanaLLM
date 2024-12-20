/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/service/service_lifetime_interface.h"

#include <memory>

#include "ksana_llm/service/inference_server.h"

namespace ksana_llm {

class KsanaServiceLifetimeManager : public ServiceLifetimeManagerInterface {
 public:
  ~KsanaServiceLifetimeManager() {}

  explicit KsanaServiceLifetimeManager(std::shared_ptr<InferenceServer> inference_server) {
    std::unique_lock<std::mutex> lock(mutex_);
    inference_server_ = inference_server;
  }

  // Stop the ksana service.
  virtual void ShutdownService() override;

 private:
  std::shared_ptr<InferenceServer> inference_server_ = nullptr;

  mutable std::mutex mutex_;
};

}  // namespace ksana_llm
