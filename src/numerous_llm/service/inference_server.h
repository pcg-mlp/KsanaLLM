/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "numerous_llm/batch_manager/batch_manager.h"
#include "numerous_llm/endpoints/endpoint.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/waiter.h"

namespace numerous_llm {

class InferenceServer {
 public:
  InferenceServer() { Initialize(); }

  ~InferenceServer();

  // Start the rpc service.
  Status StartServer();

  // Stop the rpc service.
  Status StopServer();

  // Start the handler loop.
  Status StartHandler();

  // Handle one request.
  Status HandleRequest(const Request &req);

  // Initialize inference server:
  // load weights & register model instance & start rpc port.
  Status Initialize();

 private:
  // The endpoint of this service.
  std::shared_ptr<Endpoint> endpoint_ = nullptr;

  // Global context for inference
  std::shared_ptr<Context> context_ = nullptr;

  // The global block manager.
  BlockManager *block_manager_ = nullptr;

  // The batch manager for the whole inference.
  std::shared_ptr<BatchManager> batch_manager_ = nullptr;

  // The model instances this service support.
  std::vector<std::shared_ptr<ModelInstance>> model_instances_;

  // Whether the handle loop terminated.
  std::atomic<bool> terminated_ = false;

  // channel for endpoint and inference server
  Channel<std::pair<Status, Request>> requests_queue_;
};

}  // namespace numerous_llm
