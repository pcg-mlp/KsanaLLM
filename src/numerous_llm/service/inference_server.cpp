/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <iostream>
#include <memory>

#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/endpoints/endpoint.h"
#include "numerous_llm/service/inference_server.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/memory_utils.h"
#include "numerous_llm/utils/singleton.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

InferenceServer::~InferenceServer() {
  if (block_manager_) {
    delete block_manager_;
    block_manager_ = nullptr;
  }
}

Status InferenceServer::Initialize() {
  std::shared_ptr<Environment> env = Singleton<Environment>::GetInstance();
  if (!env) {
    return Status(RET_INVALID_ARGUMENT, "The Environment is nullptr.");
  }

  context_.reset(new Context(env->GetTensorParallelSize(), env->GetPipeLineParallelSize()));

  // Initialize global block manager.
  BlockManagerConfig block_manager_config;
  Status status = env->GetBlockManagerConfig(block_manager_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get block manager config error:" + status.ToString());
  }
  block_manager_ = new BlockManager(block_manager_config, context_);
  SetBlockManager(block_manager_);

  BatchManagerConfig batch_manager_config;
  status = env->GetBatchManagerConfig(batch_manager_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get batch manager config error:" + status.ToString());
  }
  batch_manager_ = std::make_shared<BatchManager>(batch_manager_config, context_);

  // Load model instances.
  std::vector<ModelConfig> model_configs;
  status = env->GetModelList(model_configs);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get model list error:" + status.ToString());
  }
  NLLM_LOG_INFO << "Get model instance size: " << model_configs.size();

  for (const ModelConfig &model_config : model_configs) {
    std::shared_ptr<ModelInstance> model_instance = std::make_shared<ModelInstance>(model_config, context_);
    model_instance->Load();

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
  NLLM_LOG_INFO << "Handle request id " << req.req_id << ", batch size " << req.tokens.size();
  Status handle_req_status = batch_manager_->Enqueue(req.req_id, req.model_name, req.tokens, req.sampling_configs);
  if (!handle_req_status.OK()) {
    return handle_req_status;
  }
  handle_req_status = batch_manager_->WaitDone(rsp.req_id, rsp.tokens);
  return handle_req_status;
}

void InferenceServer::PrepareRespone(Status infer_status, Response &rsp) {
  std::lock_guard<std::mutex> guard(response_container_mutex_);
  response_container_[rsp.req_id] = std::make_pair<Status, Response>(std::move(infer_status), std::move(rsp));
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

    Response rsp;
    rsp.tokens.resize(req.tokens.size());
    rsp.req_id = req.req_id;
    Status infer_status = HandleRequest(req, rsp);
    PrepareRespone(infer_status, rsp);
    req.waiter->Notify();
  }

  return Status();
}

Status InferenceServer::StartServer() {
  // Start batch manager.
  batch_manager_->Start();

  // Start endpoint.
  endpoint_->Listen(requests_queue_, response_container_mutex_, response_container_);

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
