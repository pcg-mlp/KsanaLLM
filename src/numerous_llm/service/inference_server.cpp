/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/inference_server/inference_server.h"
#include "numerous_llm/endpoints/endpoint.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

InferenceServer::~InferenceServer() {
  // Stop handle new request
  Stats status = StopHandler();
  if (!status.OK()) {
    NLLM_LOG_ERROR << "Stop inference handle error:" << status.ToString();
  }

  // Wait all request done.
  status = batch_manager_->WaitAllDone();
  if (!status.OK()) {
    NLLM_LOG_ERROR << "Wait all requests done error:" << status.ToString();
  }

  // Stop rpc sercie.
  status = Stop();
  if (!status.OK()) {
    NLLM_LOG_ERROR << "Stop inference server error:" << status.ToString();
  }
}

Status InferenceServer::Initialize(std::shared_ptr<Environment> env) {
  if (!env) {
    return Status(RET_INVALID_ARGUMENT, "The Environment is nullptr.");
  }

  BatchManagerConfig batch_manager_config;
  Status status = env->GetBatchManagerConfig(batch_manager_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT,
                  "Get batch manager config error:" + status.ToString());
  }
  batch_manager_ = std::make_shared<BatchManager>(batch_manager_config);

  // Load model instances.
  std::vector<ModelConfig> model_configs;
  status = env->GetModelList(model_configs);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT,
                  "Get model list error:" + status.ToString());
  }

  for (const ModelConfig &model_config : model_configs) {
    ModelInstance model_instance;
    model_instance.Load(model_config);

    // Register model instance.
    model_instances_.push_back(model_instance);
    batch_manager_->RegisterModelInstance(model_instance);
  }

  EndpointConfig endpoint_config;
  status = env->GetEndpointConfig(endpoint_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT,
                  "Get endpoint config error:" + status.ToString());
  }
  endpoint_ = std::make_shared<Endpoint>(endpoint_config);

  return Status();
}

Status InferenceServer::HandleRequest(const Request &req, Response &rsp) {
  batch_manager_->Enqueue(req.req_id, req.tensor_maps, req.sampling_configs);
  return batch_manager_->WaitDone(rsp.req_id, rsp.tensor_maps);
}

} // namespace numerous_llm.
