/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "supervisor/service_lifetime_interface.h"

#include <memory>

#include "supervisor/inference_supervisor.h"

namespace ksana_llm {

class KsanaServiceLifetimeManager : public ServiceLifetimeManagerInterface {
 public:
  ~KsanaServiceLifetimeManager() {}

  explicit KsanaServiceLifetimeManager(std::shared_ptr<InferenceSupervisor> inference_supervisor) {
    std::unique_lock<std::mutex> lock(mutex_);
    inference_supervisor_ = inference_supervisor;
  }

  // Stop the ksana service.
  virtual void ShutdownService() override;

 private:
  std::shared_ptr<InferenceSupervisor> inference_supervisor_ = nullptr;

  mutable std::mutex mutex_;
};

}  // namespace ksana_llm
