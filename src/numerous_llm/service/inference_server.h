/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "numerous_llm/batch_manager/batch_manager.h"
#include "numerous_llm/endpoints/base/base_endpoint.h"
#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class InferenceServer {
public:
  explicit InferenceServer() {}
  ~InferenceServer();

  // Start and stop the rpc service.
  Status Start();
  Status Stop();

  // Stop handler, new incoming request will be dropped.
  Status StopHandler();

  // Handle one request.
  Status HandleRequest(const Request &req, Response &rsp);

private:
  // Initialize inference server:
  // load weights & register model instance & start rpc port.
  Status Initialize(std::shared_ptr<Environment> env);

private:
  // The endpoint of this service.
  std::shared_ptr<BaseEndpoint> endpoint_ = nullptr;

  // The batch manager for the whole inference.
  std::shared_ptr<BatchManager> batch_manager_ = nullptr;

  // The model instances this service support.
  std::vector<ModelInstance> model_instances_;
};

} // namespace numerous_llm
