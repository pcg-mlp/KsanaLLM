/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/service/service_lifetime.h"

namespace ksana_llm {

void KsanaServiceLifetimeManager::ShutdownService() {
  KLLM_LOG_INFO << "ShutdownService invoked.";

  std::unique_lock<std::mutex> lock(mutex_);
  if (inference_server_) {
    inference_server_->Stop();
    inference_server_ = nullptr;
  }

  // Note: The python's uvicorn have no graceful exit meothd, so exit process here.
  // All destructor will be called.
  KLLM_LOG_INFO << "ShutdownService read to exit.";
  _exit(0);
}

}  // namespace ksana_llm
