/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "supervisor/service_lifetime.h"

namespace ksana_llm {

void KsanaServiceLifetimeManager::ShutdownService() {
  KLLM_LOG_INFO << "ShutdownService invoked.";
  // TODO(winminkong): emporarily force-exit, we’ll follow up to fix the hang when invoking the destructor later.
  _exit(0);
  std::unique_lock<std::mutex> lock(mutex_);
  if (inference_supervisor_) {
    inference_supervisor_->Destroy();
    inference_supervisor_ = nullptr;
  }

  // Note: The python's uvicorn have no graceful exit meothd, so exit process here.
  // All destructor will be called.
  KLLM_LOG_INFO << "ShutdownService read to exit.";
}

}  // namespace ksana_llm
