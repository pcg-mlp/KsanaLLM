/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <unordered_map>
#include "runtime/executor_infer_request.h"
#include "utils/request.h"

namespace ksana_llm {

struct ExecutorRequest {
  // For schedule output.
  std::shared_ptr<ExecutorInferRequest> infer_request = nullptr;

  // other cached instances.
};

// Used to cache some necessary information of request in executor process.
// So that we can transfer only increase part of every request.
class ExecutorRequestManager {
 public:
  ExecutorRequestManager() {}

  void AddInferRequest(int64_t req_id, std::shared_ptr<ExecutorInferRequest> infer_req);
  void UpdateInferRequest(int64_t req_id, std::shared_ptr<InferRequestPatch> infer_req_patch);

  std::shared_ptr<ExecutorInferRequest> GetInferRequest(int64_t req_id);

  // Erase a request from cache.  // Invoked only when a request is finished.
  void EraseRequest(int64_t req_id);

 private:
  // The req_id to cached block.
  std::unordered_map<int64_t, ExecutorRequest> cached_requests_;

  // Protect global map.
  std::mutex mutex_;
};

}  // namespace ksana_llm
