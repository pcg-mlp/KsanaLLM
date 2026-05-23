/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "executor/executor_request_manager.h"
#include "fmt/format.h"
#include "utils/logger.h"

namespace ksana_llm {

void ExecutorRequestManager::AddInferRequest(int64_t req_id, std::shared_ptr<ExecutorInferRequest> infer_req) {
  std::lock_guard<std::mutex> lock(mutex_);

  // If the request is existed yet, such as recompute, just replace it.
  cached_requests_[req_id].infer_request = infer_req;
  KLLM_LOG_DEBUG << "req " << req_id << " cached." << std::endl;
}

void ExecutorRequestManager::UpdateInferRequest(int64_t req_id, std::shared_ptr<InferRequestPatch> infer_req_patch) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (cached_requests_.find(req_id) == cached_requests_.end()) {
    KLLM_THROW(fmt::format("Update req {} error, req not cached yet.", req_id));
  }

  auto& executor_request = cached_requests_[req_id];
  size_t forwarding_tokens_offset = infer_req_patch->forwarding_tokens_offset;
  executor_request.infer_request->infer_stage = infer_req_patch->infer_stage;
  executor_request.infer_request->step = infer_req_patch->step;
  executor_request.infer_request->forwarding_tokens_draft_num = infer_req_patch->forwarding_tokens_draft_num;
  executor_request.infer_request->kv_cached_token_num = infer_req_patch->kv_cached_token_num;
  executor_request.infer_request->prefix_cache_len = infer_req_patch->prefix_cache_len;
  executor_request.infer_request->sampling_token_num = infer_req_patch->sampling_token_num;

  // TODO(yancyliu): transfer full forwarding_tokens & kv_cache_blocks now, change to incremental later.
  executor_request.infer_request->forwarding_tokens = infer_req_patch->delta_forwarding_tokens;
  executor_request.infer_request->kv_cache_blocks = infer_req_patch->delta_kv_cache_blocks;

  // Transfer full swa_kv_cache_blocks (SWA blocks are always sent in full).
  executor_request.infer_request->swa_kv_cache_blocks = infer_req_patch->swa_kv_cache_blocks;
  executor_request.infer_request->swa_kv_cache_block_idx_offsets = infer_req_patch->swa_kv_cache_block_idx_offsets;

  KLLM_LOG_DEBUG << "req " << req_id << " updated." << std::endl;
}

std::shared_ptr<ExecutorInferRequest> ExecutorRequestManager::GetInferRequest(int64_t req_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (cached_requests_.find(req_id) != cached_requests_.end()) {
    return cached_requests_.at(req_id).infer_request;
  }
  return nullptr;
}

void ExecutorRequestManager::EraseRequest(int64_t req_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  cached_requests_.erase(req_id);
  KLLM_LOG_DEBUG << "req " << req_id << " erased." << std::endl;
}

}  // namespace ksana_llm
