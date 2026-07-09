/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/global_cache_query.h"

#include <chrono>
#include <thread>

#include "configure/environment.h"
#include "ipc/core/ipc_func_serialization.h"
#include "ipc/ipc_func_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

Status GlobalCacheQueryRequest::Serialize(void* ptr, size_t* bytes) const {
  size_t offset = 0;
  SerializePod<int64_t>(req_id, ptr, offset);
  SerializeVector<int>(token_ids, ptr, offset);
  SerializePod<size_t>(local_cached_len, ptr, offset);
  SerializePod<int>(block_size, ptr, offset);
  SerializeVector<int>(kv_cache_block_ids, ptr, offset);
  *bytes = offset;
  return Status();
}

Status GlobalCacheQueryRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  req_id = DeserializePod<int64_t>(ptr, offset);
  DeserializeVector<int>(token_ids, ptr, offset);
  local_cached_len = DeserializePod<size_t>(ptr, offset);
  block_size = DeserializePod<int>(ptr, offset);
  DeserializeVector<int>(kv_cache_block_ids, ptr, offset);
  return Status();
}

Status GlobalCacheQueryCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const GlobalCacheQueryRequest* real_req = reinterpret_cast<const GlobalCacheQueryRequest*>(request);
  // 仅 V1 "decode" role 触发 retry-until-full-hit + token retry. pd_v2_decode 走 standalone
  // 风格的单次查询: peer Prefill 通过 RDMA 兜底 partial / miss, 不需要在这里阻塞重试.
  const bool is_decode_role = (role_ == "decode");

  KLLM_LOG_SCHEDULER << "device_id=" << device_id_ << " received global cache query"
                     << ", req_id=" << real_req->req_id << ", token_num=" << real_req->token_ids.size()
                     << ", local_cached_len=" << real_req->local_cached_len
                     << ", role=" << (role_.empty() ? "standalone" : role_);

  std::vector<int> matched_ranges;
  bool success = false;
  std::vector<int> cached_tokens;
  bool has_cached_tokens = false;

  // 上报逻辑封装为 lambda，所有退出路径统一调用
  auto report = [&]() -> Status {
    KLLM_LOG_DEBUG << "reporting, device_id=" << device_id_ << ", req_id=" << real_req->req_id
                   << ", success=" << success << ", matched_ranges_size=" << matched_ranges.size()
                   << ", has_cached_tokens=" << has_cached_tokens << ", cached_tokens_count=" << cached_tokens.size()
                   << ", role=" << (role_.empty() ? "standalone" : role_);

    if (ipc_func_wrapper_) {
      Status report_status = ipc_func_wrapper_->ReportGlobalCacheQueryDone(real_req->req_id, device_id_, matched_ranges,
                                                                           success, cached_tokens, has_cached_tokens);
      if (!report_status.OK()) {
        KLLM_LOG_ERROR << "report done failed, req_id=" << real_req->req_id;
      }
    }
    return Status();
  };

  if (!connector_) {
    KLLM_LOG_WARNING << "connector not set, device_id=" << device_id_;
    return report();
  }

  if (!block_allocator_group_) {
    block_allocator_group_ = block_allocator_manager_->GetBlockAllocatorGroup(dp_group_id_);
  }

  std::vector<void*> block_ptrs;
  if (block_allocator_group_) {
    auto device_allocator = block_allocator_group_->GetDeviceBlockAllocator(device_id_);
    if (device_allocator) {
      device_allocator->GetBlockPtrs(real_req->kv_cache_block_ids, block_ptrs);
    }
  }

  if (block_ptrs.empty()) {
    KLLM_LOG_WARNING << "failed to get block pointers, device_id=" << device_id_;
    return report();
  }

  int tp_size = 1;
  int tp_rank = 0;
  size_t block_byte_size = block_allocator_group_->GetBlockAllocatorGroupConfig().block_size;
  int total_tokens = static_cast<int>(real_req->token_ids.size());
  int local_cached = static_cast<int>(real_req->local_cached_len);

  // decode 角色：QueryAndFetch 内部自动重试未命中 block，已拉取的不会重复拷贝
  // standalone / prefill 角色：单次查询
  KLLM_LOG_DEBUG << "QueryAndFetch, device_id=" << device_id_ << ", req_id=" << real_req->req_id
                 << ", token_num=" << real_req->token_ids.size() << ", local_cached_len=" << real_req->local_cached_len
                 << ", tp_size=" << tp_size << ", tp_rank=" << tp_rank << ", layer_index=" << 0
                 << ", block_size=" << real_req->block_size << ", block_byte_size=" << block_byte_size
                 << ", block_ptrs_size=" << block_ptrs.size() << ", retry=" << is_decode_role
                 << ", timeout_ms=" << timeout_ms_;
  Status status =
      connector_->QueryAndFetch(real_req->token_ids, real_req->local_cached_len, tp_size, tp_rank,
                                /*layer_index=*/0, real_req->block_size, block_byte_size, block_ptrs, matched_ranges,
                                /*retry_until_full_hit=*/is_decode_role, timeout_ms_);
  success = status.OK();
  if (!success) {
    KLLM_LOG_WARNING << "QueryAndFetch failed, device_id=" << device_id_ << ", req_id=" << real_req->req_id
                     << ", error=" << status.GetMessage();
  }

  KLLM_LOG_DEBUG << "QueryAndFetch result, device_id=" << device_id_ << ", req_id=" << real_req->req_id
                 << ", success=" << success << ", matched_ranges_size=" << matched_ranges.size()
                 << ", is_decode_role=" << is_decode_role;

  // 仅卡 0 查询缓存的生成 token(s)
  if (success && device_id_ == 0 && !matched_ranges.empty()) {
    bool all_blocks_hit = IsFullHit(matched_ranges, total_tokens, local_cached);
    KLLM_LOG_DEBUG << "device_id=" << device_id_ << ", req_id=" << real_req->req_id
                   << ", all_blocks_hit=" << all_blocks_hit << ", total_tokens=" << total_tokens
                   << ", local_cached=" << local_cached;

    if (is_decode_role && all_blocks_hit) {
      // decode 角色：KV 已全命中，重试查询 token 直到找到或剩余时间耗尽
      auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);
      constexpr int kRetryIntervalMs = 10;

      while (std::chrono::steady_clock::now() < deadline) {
        connector_->QueryFirstTokens(real_req->token_ids, cached_tokens, has_cached_tokens);
        if (has_cached_tokens) {
          KLLM_LOG_SCHEDULER << "decode mode, tokens found"
                             << ", device_id=" << device_id_ << ", req_id=" << real_req->req_id
                             << ", token_count=" << cached_tokens.size();
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryIntervalMs));
      }

      if (!has_cached_tokens) {
        KLLM_LOG_WARNING << "decode mode, tokens not found after timeout"
                         << ", device_id=" << device_id_ << ", req_id=" << real_req->req_id;
      }
    } else if (role_.empty() && all_blocks_hit) {
      // standalone 模式：顺带查询 token（不重试）
      connector_->QueryFirstTokens(real_req->token_ids, cached_tokens, has_cached_tokens);
    }
  }

  return report();
}

bool GlobalCacheQueryCallable::IsFullHit(const std::vector<int>& matched_ranges, int total_tokens,
                                         int local_cached_len) {
  if (matched_ranges.empty() || total_tokens <= 0) return false;
  // 从 local_cached_len 开始检查是否有连续覆盖到 total_tokens-1
  int pos = local_cached_len;
  for (size_t i = 0; i + 1 < matched_ranges.size(); i += 2) {
    if (matched_ranges[i] > pos) return false;
    pos = std::max(pos, matched_ranges[i + 1] + 1);
  }
  return pos >= total_tokens;
}

void GlobalCacheQueryCallable::SetGlobalCacheConnector(std::shared_ptr<GlobalCacheConnectorInterface> connector) {
  connector_ = connector;
}

void GlobalCacheQueryCallable::SetBlockAllocatorManager(
    std::shared_ptr<BlockAllocatorManagerInterface> block_allocator_manager, int dp_group_id) {
  block_allocator_manager_ = block_allocator_manager;
  dp_group_id_ = dp_group_id;
}

void GlobalCacheQueryCallable::SetIpcFuncWrapper(IpcFuncWrapper* ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
}

void GlobalCacheQueryCallable::SetEnvironment(std::shared_ptr<Environment> env) { env_ = env; }

void GlobalCacheQueryCallable::SetDeviceId(int device_id) { device_id_ = device_id; }

void GlobalCacheQueryCallable::SetGlobalCacheRole(const std::string& role, int timeout_ms) {
  role_ = role;
  timeout_ms_ = timeout_ms;
  KLLM_LOG_INFO << "role=" << (role_.empty() ? "(standalone)" : role_) << ", timeout_ms=" << timeout_ms_
                << ", device_id=" << device_id_;
}

}  // namespace ksana_llm
