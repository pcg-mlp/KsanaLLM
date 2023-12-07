/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "numerous_llm/batch_manager/batch_manager.h"
#include "numerous_llm/endpoints/endpoint.h"
#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class InferenceServer {
 public:
  // Start the rpc service.
  Status StartServer();

  // Stop the rpc service.
  Status StopServer();

  // Start the handler loop.
  Status StartHandler();

  // Handle one request.
  Status HandleRequest(const Request &req, Response &rsp);

  // Initialize inference server:
  // load weights & register model instance & start rpc port.
  Status Initialize(std::shared_ptr<Environment> env);

 private:
  // The endpoint of this service.
  std::shared_ptr<Endpoint> endpoint_ = nullptr;

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
