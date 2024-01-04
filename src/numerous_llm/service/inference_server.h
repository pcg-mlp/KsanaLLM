/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "numerous_llm/endpoints/base/base_endpoint.h"
#include "numerous_llm/service/inference_engine.h"
#include "numerous_llm/utils/channel.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/waiter.h"

namespace numerous_llm {

class InferenceServer {
 public:
  InferenceServer();

  ~InferenceServer() {}

  // Start the inference server.
  Status Start();

  // Stop the inference server.
  Status Stop();

 private:
  // Wait until serve stopped.
  Status WaitUntilStop();

 private:
  // The inference engine.
  std::shared_ptr<InferenceEngine> inference_engine_ = nullptr;

  // The rpc endpoint of this service.
  std::shared_ptr<RpcEndpoint> endpoint_ = nullptr;

  // channel for endpoint and inference server
  Channel<std::pair<Status, Request>> request_queue_;

  // Use to gracefully stopped.
  std::shared_ptr<Waiter> waiter_ = nullptr;
};

}  // namespace numerous_llm
