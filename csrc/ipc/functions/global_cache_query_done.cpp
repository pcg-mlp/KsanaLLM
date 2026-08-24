/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/global_cache_query_done.h"

#include "batch_scheduler/global_cache_fetcher.h"
#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"

namespace ksana_llm {

Status GlobalCacheQueryDoneRequest::Serialize(void* ptr, size_t* bytes) const {
  size_t offset = 0;
  SerializePod<int64_t>(req_id, ptr, offset);
  SerializePod<int>(device_id, ptr, offset);
  SerializeVector<int>(matched_ranges, ptr, offset);
  SerializePod<bool>(success, ptr, offset);
  SerializeVector<int>(cached_tokens, ptr, offset);
  SerializePod<bool>(has_cached_tokens, ptr, offset);
  SerializePod<int64_t>(wait_us, ptr, offset);
  SerializePod<int64_t>(qf_us, ptr, offset);
  SerializePod<int64_t>(token_us, ptr, offset);
  SerializePod<int64_t>(report_us, ptr, offset);
  SerializePod<int64_t>(call_us, ptr, offset);
  SerializePod<int64_t>(report_wall_us, ptr, offset);
  SerializePod<size_t>(swa_aligned_hit_len, ptr, offset);
  SerializeVector<size_t>(swa_filled_block_idxs, ptr, offset);
  *bytes = offset;
  return Status();
}

Status GlobalCacheQueryDoneRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  req_id = DeserializePod<int64_t>(ptr, offset);
  device_id = DeserializePod<int>(ptr, offset);
  DeserializeVector<int>(matched_ranges, ptr, offset);
  success = DeserializePod<bool>(ptr, offset);
  DeserializeVector<int>(cached_tokens, ptr, offset);
  has_cached_tokens = DeserializePod<bool>(ptr, offset);
  wait_us = DeserializePod<int64_t>(ptr, offset);
  qf_us = DeserializePod<int64_t>(ptr, offset);
  token_us = DeserializePod<int64_t>(ptr, offset);
  report_us = DeserializePod<int64_t>(ptr, offset);
  call_us = DeserializePod<int64_t>(ptr, offset);
  report_wall_us = DeserializePod<int64_t>(ptr, offset);
  swa_aligned_hit_len = DeserializePod<size_t>(ptr, offset);
  DeserializeVector<size_t>(swa_filled_block_idxs, ptr, offset);
  return Status();
}

Status GlobalCacheQueryDoneCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const GlobalCacheQueryDoneRequest* real_req = reinterpret_cast<const GlobalCacheQueryDoneRequest*>(request);

  KLLM_LOG_SCHEDULER << "received done notification, req_id=" << real_req->req_id
                     << ", device_id=" << real_req->device_id << ", success=" << real_req->success
                     << ", matched_ranges_size=" << real_req->matched_ranges.size();

  if (fetcher_) {
    GlobalCacheQueryResult result;
    result.req_id = real_req->req_id;
    result.device_id = real_req->device_id;
    result.matched_ranges = real_req->matched_ranges;
    result.success = real_req->success;
    result.cached_tokens = real_req->cached_tokens;
    result.has_cached_tokens = real_req->has_cached_tokens;
    result.wait_us = real_req->wait_us;
    result.qf_us = real_req->qf_us;
    result.token_us = real_req->token_us;
    result.report_us = real_req->report_us;
    result.call_us = real_req->call_us;
    result.report_wall_us = real_req->report_wall_us;
    result.swa_aligned_hit_len = real_req->swa_aligned_hit_len;
    result.swa_filled_block_idxs = real_req->swa_filled_block_idxs;
    fetcher_->OnDeviceComplete(result);
  } else {
    KLLM_LOG_WARNING << "fetcher not set";
  }

  return Status();
}

void GlobalCacheQueryDoneCallable::SetGlobalCacheFetcher(std::shared_ptr<GlobalCacheFetcher> fetcher) {
  fetcher_ = fetcher;
}

}  // namespace ksana_llm
