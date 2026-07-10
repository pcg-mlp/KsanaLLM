/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/script.h>
#include <memory>
#include <string>

#include "configure/environment.h"
#include "supervisor/inference_supervisor.h"
#include "utils/status.h"

namespace ksana_llm {

// The torch OP for inference.
class ServingOp : public torch::jit::CustomClassHolder {
 public:
  ServingOp(const std::string &config_file, bool is_engine, int executor_id,
            const std::string &json_model_override_args = "");
  ~ServingOp();

  // Initialize ipc resources.
  void InitIpc();

  // Initialize the serving server.
  void InitServing(int think_end_token_id = -1);

  // Start the serving executor, never return.
  void StartExecutor();

  // Generate a response.
  Status Generate(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                  const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                  ksana_llm::KsanaPythonOutput &ksana_python_output);

  // Generate a response, in streaming mode.
  Status GenerateStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                           const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                           std::shared_ptr<StreamingIterator> &streaming_iterator);

  /**
   * Generate responses for requests toward the forward interface.
   * This interface performs a single-step inference, i.e., the context phase.
   * It allows a batch of requests to be submitted at once, and each request
   * can specify multiple targets to obtain model outputs such as transformer,
   * final layer norm, logits, etc.
   * Refer to serving_forward_client for more details.
   */
  Status Forward(const std::string &request_bytes,
                 const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                 std::string &response_bytes);

  // Health: whether this (Decode) engine sees at least one healthy pd_v2
  // Prefill peer. False when this process is not a pd_v2 Decode node.
  // Exposed to Python so the /health endpoint can gate on Prefill reachability.
  bool HasHealthyPrefillPeer() const;

 private:
  // Set logger name.
  void SetProcessLogger(const std::string &role, int device_id = -1);

 public:
  std::string config_file_;
  bool is_engine_ = false;
  int executor_id_ = -1;

  // The config of endpoint.
  EndpointConfig endpoint_config_;

 private:
  // The inference supervisor.
  std::shared_ptr<InferenceSupervisor> inference_supervisor_ = nullptr;
};

}  // namespace ksana_llm
