/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <iostream>

#include "numerous_llm/endpoints/endpoint.h"
#include "numerous_llm/service/inference_server.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Status InferenceServer::Initialize(std::shared_ptr<Environment> env) {
  if (!env) {
    return Status(RET_INVALID_ARGUMENT, "The Environment is nullptr.");
  }

  BatchManagerConfig batch_manager_config;
  Status status = env->GetBatchManagerConfig(batch_manager_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get batch manager config error:" + status.ToString());
  }
  batch_manager_ = std::make_shared<BatchManager>(batch_manager_config);

  // Load model instances.
  std::vector<ModelConfig> model_configs;
  status = env->GetModelList(model_configs);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get model list error:" + status.ToString());
  }
  NLLM_LOG_INFO << "Get model instance size: " << model_configs.size();

  for (const ModelConfig &model_config : model_configs) {
    std::shared_ptr<ModelInstance> model_instance = std::make_shared<ModelInstance>();
    model_instance->Load(model_config);

    // Register model instance.
    model_instances_.push_back(model_instance);
    batch_manager_->RegisterModelInstance(model_instance);
  }

  EndpointConfig endpoint_config;
  status = env->GetEndpointConfig(endpoint_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get endpoint config error:" + status.ToString());
  }
  endpoint_ = std::make_shared<Endpoint>(endpoint_config);

  return Status();
}

Status InferenceServer::HandleRequest(const Request &req, Response &rsp) {
  batch_manager_->Enqueue(req.req_id, req.tensor_maps, req.sampling_configs);
  return batch_manager_->WaitDone(rsp.req_id, rsp.tensor_maps);
}

Status InferenceServer::StartHandler() {
  NLLM_LOG_INFO << "Start handler";

  while (!terminated_) {
    Request req;
    std::pair<Status, Request> req_pair;

    requests_queue_.Read(&req_pair);

    Status status = req_pair.first;
    req = req_pair.second;

    if (status.GetCode() == RET_TERMINATED) {
      break;
    }

    // TODO: should not block.
    Response rsp;
    HandleRequest(req, rsp);

    endpoint_->Send(rsp);
  }

  return Status();
}

Status InferenceServer::StartServer() {
  // Start batch manager.
  batch_manager_->Start();

  // Start endpoint.
  endpoint_->Listen(requests_queue_);

  // Start service handler.
  StartHandler();

  return Status();
}

Status InferenceServer::StopServer() {
  NLLM_LOG_INFO << "Recive stop signal, ready to quit.";
  if (terminated_) {
    return Status();
  }

  terminated_ = true;

  // Wait all request done.
  NLLM_LOG_INFO << "Waiting all running request.";
  Status status = batch_manager_->WaitAllDone();
  if (!status.OK()) {
    NLLM_LOG_ERROR << "Wait all requests done error:" << status.ToString();
  }

  // Close endpoint.
  NLLM_LOG_INFO << "Stop endpoint.";
  endpoint_->Close();

  // Stop the batch manger.
  NLLM_LOG_INFO << "Stop batch manager.";
  batch_manager_->Stop();
  return Status();
}

}  // namespace numerous_llm.
