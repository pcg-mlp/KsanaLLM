/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

#include "configure/environment.h"
#include "executor/inference_executor.h"
#include "ipc/ipc_func_wrapper.h"
#include "service/inference_service.h"

namespace ksana_llm {

// Used to manage parent and children processes.
class InferenceSupervisor {
 public:
  InferenceSupervisor(const std::string& config_file, bool is_engine, int executor_id,
                      const std::string& json_model_override_args = "");
  ~InferenceSupervisor();

  // Initialize ipc resources.
  Status InitIpc();

  // Initialize the inference service.
  Status InitService(const EndpointConfig& endpoint_config, int think_end_token_id = -1);

  // Initialize the inference excutor.
  Status InitExecutor();

  // Destroy the inference supervisor.
  // This method should close all child processes, and stop all extra threads.
  Status Destroy();

  // Get the inferenc service.
  std::shared_ptr<InferenceService>& GetService();

  // Health: whether this node sees at least one healthy pd_v2 Prefill peer.
  // False when this process is not a pd_v2 Decode node or the service isn't up.
  bool HasHealthyPrefillPeer() const;

 private:
  std::string config_file_;
  bool is_engine_ = false;
  int executor_id_ = -1;

  EndpointConfig endpoint_config_;

  // The global environment & context, created only once.
  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;

  // The server part.
  std::shared_ptr<InferenceService> inference_service_ = nullptr;

  // The executor part.
  std::shared_ptr<InferenceExecutor> inference_executor_ = nullptr;

  // The ipc func wrapper, used to invoke functions between processes.
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_ = nullptr;
};

}  // namespace ksana_llm
