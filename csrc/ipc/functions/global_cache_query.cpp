/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/global_cache_query.h"

#include <algorithm>
#include <chrono>
#include <thread>

#include "configure/environment.h"
#include "ipc/core/ipc_func_serialization.h"
#include "ipc/ipc_func_wrapper.h"
#include "profiler/timer.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

Status GlobalCacheQueryRequest::Serialize(void* ptr, size_t* bytes) const {
  size_t offset = 0;
  SerializePod<int64_t>(req_id, ptr, offset);
  SerializeVector<int>(token_ids, ptr, offset);
  SerializePod<size_t>(local_cached_len, ptr, offset);
  SerializePod<int>(block_size, ptr, offset);
  SerializeVector<int>(kv_cache_block_ids, ptr, offset);
  SerializePod<int64_t>(enqueue_wall_us, ptr, offset);
  SerializeVector<int>(swa_kv_cache_block_ids, ptr, offset);
  SerializeVector<size_t>(swa_block_idx_offsets, ptr, offset);
  SerializePod<size_t>(swa_window_block_num, ptr, offset);
  SerializePod<size_t>(max_cached_len, ptr, offset);
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
  enqueue_wall_us = DeserializePod<int64_t>(ptr, offset);
  DeserializeVector<int>(swa_kv_cache_block_ids, ptr, offset);
  DeserializeVector<size_t>(swa_block_idx_offsets, ptr, offset);
  swa_window_block_num = DeserializePod<size_t>(ptr, offset);
  max_cached_len = DeserializePod<size_t>(ptr, offset);
  return Status();
}

Status GlobalCacheQueryCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const GlobalCacheQueryRequest* real_req = reinterpret_cast<const GlobalCacheQueryRequest*>(request);
  // 仅 V1 "decode" role 触发 retry-until-full-hit + token retry. pd_v2_decode 走 standalone
  // 风格的单次查询: peer Prefill 通过 RDMA 兜底 partial / miss, 不需要在这里阻塞重试.
  const bool is_decode_role = (role_ == "decode");
  const int64_t call_start_us = ProfileTimer::GetCurrentTimeInUs();
  // enqueue → Call：含 IPC 送达 + long-task 队列 + heavy mutex 等待
  const int64_t wait_us =
      (real_req->enqueue_wall_us > 0) ? (call_start_us - real_req->enqueue_wall_us) : static_cast<int64_t>(-1);

  KLLM_LOG_SCHEDULER << "device_id=" << device_id_ << " received global cache query"
                     << ", req_id=" << real_req->req_id << ", token_num=" << real_req->token_ids.size()
                     << ", local_cached_len=" << real_req->local_cached_len
                     << ", role=" << (role_.empty() ? "standalone" : role_) << ", wait_us=" << wait_us;

  std::vector<int> matched_ranges;
  bool success = false;
  std::vector<int> cached_tokens;
  bool has_cached_tokens = false;
  GlobalCacheSwaResult swa_result;
  int64_t qf_us = -1;
  int64_t token_us = 0;
  int64_t report_us = -1;
  int64_t call_us = -1;

  // 上报逻辑封装为 lambda，所有退出路径统一调用
  auto report = [&]() -> Status {
    call_us = ProfileTimer::GetCurrentTimeInUs() - call_start_us;
    const int64_t report_wall_us = ProfileTimer::GetCurrentTimeInUs();
    const int64_t report_start_us = report_wall_us;

    KLLM_LOG_DEBUG << "reporting, device_id=" << device_id_ << ", req_id=" << real_req->req_id
                   << ", success=" << success << ", matched_ranges_size=" << matched_ranges.size()
                   << ", has_cached_tokens=" << has_cached_tokens << ", cached_tokens_count=" << cached_tokens.size()
                   << ", role=" << (role_.empty() ? "standalone" : role_);

    if (ipc_func_wrapper_) {
      Status report_status = ipc_func_wrapper_->ReportGlobalCacheQueryDone(
          real_req->req_id, device_id_, matched_ranges, success, cached_tokens, has_cached_tokens, wait_us, qf_us,
          // report_us 在 Invoke 返回前未知，消息里固定传 0；真实耗时仅记本地 report_us 日志。
          token_us, /*report_us=*/0, call_us, report_wall_us, swa_result.swa_aligned_hit_len,
          swa_result.filled_block_idxs);
      report_us = ProfileTimer::GetCurrentTimeInUs() - report_start_us;
      if (!report_status.OK()) {
        KLLM_LOG_ERROR << "report done failed, req_id=" << real_req->req_id;
      }
    } else {
      report_us = 0;
    }

    KLLM_LOG_SCHEDULER << "query_card done, req_id=" << real_req->req_id << " device=" << device_id_
                       << " wait_us=" << wait_us << " qf_us=" << qf_us << " token_us=" << token_us
                       << " report_us=" << report_us << " call_us=" << call_us << " success=" << success
                       << " matched_ranges=" << matched_ranges.size() << " tokens=" << real_req->token_ids.size()
                       << " local_cached=" << real_req->local_cached_len
                       << " swa_aligned_hit=" << swa_result.swa_aligned_hit_len
                       << " swa_filled=" << swa_result.filled_block_idxs.size();
    return Status();
  };

  if (!connector_) {
    KLLM_LOG_WARNING << "connector not set, device_id=" << device_id_;
    return report();
  }

  if (!block_allocator_group_) {
    // 多 long-task 线程可能并发进入 Call，懒加载只需初始化一次
    std::call_once(block_allocator_group_once_, [this]() {
      if (block_allocator_manager_) {
        block_allocator_group_ = block_allocator_manager_->GetBlockAllocatorGroup(dp_group_id_);
      }
    });
  }

  std::vector<void*> block_ptrs;
  if (block_allocator_group_) {
    auto device_allocator = block_allocator_group_->GetDeviceBlockAllocator(device_id_);
    if (device_allocator) {
      // GetBlockPtrs：block_id 必须已在 executor 池的 allocated_blocks_ 中，否则直接失败。
      Status ptr_status = device_allocator->GetBlockPtrs(real_req->kv_cache_block_ids, block_ptrs);
      if (!ptr_status.OK()) {
        KLLM_LOG_ERROR << "query_block_pool_miss device=" << device_id_ << " req_id=" << real_req->req_id
                       << " wait_us=" << wait_us << " err=" << ptr_status.GetMessage()
                       << " n_ids=" << real_req->kv_cache_block_ids.size()
                       << " ids=" << Vector2Str(real_req->kv_cache_block_ids)
                       << " tokens=" << real_req->token_ids.size() << " local_cached=" << real_req->local_cached_len
                       << " block_size=" << real_req->block_size << " skip_query=1";
        return report();
      }
    }
  }

  if (block_ptrs.empty()) {
    KLLM_LOG_ERROR << "query_block_ptrs_empty device=" << device_id_ << " req_id=" << real_req->req_id
                   << " wait_us=" << wait_us << " n_ids=" << real_req->kv_cache_block_ids.size()
                   << " ids=" << Vector2Str(real_req->kv_cache_block_ids) << " skip_query=1";
    return report();
  }
  if (block_ptrs.size() != real_req->kv_cache_block_ids.size()) {
    KLLM_LOG_ERROR << "query_block_ptr_size_mismatch device=" << device_id_ << " req_id=" << real_req->req_id
                   << " n_ids=" << real_req->kv_cache_block_ids.size() << " n_ptrs=" << block_ptrs.size()
                   << " ids=" << Vector2Str(real_req->kv_cache_block_ids) << " skip_query=1";
    return report();
  }

  int tp_size = 1;
  int tp_rank = 0;
  size_t block_byte_size = block_allocator_group_->GetBlockAllocatorGroupConfig().block_size;
  int total_tokens = static_cast<int>(real_req->token_ids.size());
  int local_cached = static_cast<int>(real_req->local_cached_len);

  // SWA 上下文：只有拿到 SWA block 指针才启用对齐相位。缺任何一项就退回纯主 KV 行为，
  // 此时 swa_aligned_hit_len 保持 0，fetcher 会按「无 SWA 对齐结果」处理。
  GlobalCacheSwaContext swa_ctx;
  bool use_swa = swa_block_allocator_manager_ != nullptr && !real_req->swa_kv_cache_block_ids.empty() &&
                 real_req->swa_kv_cache_block_ids.size() == real_req->swa_block_idx_offsets.size();
  if (use_swa) {
    std::call_once(swa_block_allocator_group_once_, [this]() {
      swa_block_allocator_group_ = swa_block_allocator_manager_->GetBlockAllocatorGroup(dp_group_id_);
    });
    auto swa_device_allocator =
        swa_block_allocator_group_ ? swa_block_allocator_group_->GetDeviceBlockAllocator(device_id_) : nullptr;
    if (swa_device_allocator) {
      // 失败则清空 ptrs，走下方 incomplete 回退（仍只做主 KV），不改变成功路径。
      Status swa_ptr_status =
          swa_device_allocator->GetBlockPtrs(real_req->swa_kv_cache_block_ids, swa_ctx.swa_block_ptrs);
      if (!swa_ptr_status.OK()) {
        KLLM_LOG_WARNING << "swa GetBlockPtrs failed, fall back to main-kv only, device_id=" << device_id_
                         << ", req_id=" << real_req->req_id << ", err=" << swa_ptr_status.GetMessage();
        swa_ctx.swa_block_ptrs.clear();
      }
    }
    swa_ctx.local_swa_block_idxs = real_req->swa_block_idx_offsets;
    swa_ctx.swa_block_byte_size =
        swa_block_allocator_group_ ? swa_block_allocator_group_->GetBlockAllocatorGroupConfig().block_size : 0;
    swa_ctx.swa_window_block_num = real_req->swa_window_block_num;
    swa_ctx.max_hit_token_num = real_req->max_cached_len;
    swa_ctx.req_id = real_req->req_id;

    use_swa = swa_ctx.swa_block_ptrs.size() == swa_ctx.local_swa_block_idxs.size() && swa_ctx.swa_block_byte_size > 0 &&
              swa_ctx.max_hit_token_num > 0;
    if (!use_swa) {
      // max_cached_len=0 多见于 health check / 极短序列，属预期路径，不必 WARNING。
      if (swa_ctx.max_hit_token_num == 0) {
        KLLM_LOG_DEBUG << "swa context incomplete (no seq upper bound), fall back to main-kv only, device_id="
                       << device_id_ << ", req_id=" << real_req->req_id;
      } else {
        KLLM_LOG_WARNING << "swa context incomplete, fall back to main-kv only, device_id=" << device_id_
                         << ", req_id=" << real_req->req_id << ", ptrs=" << swa_ctx.swa_block_ptrs.size()
                         << ", idxs=" << swa_ctx.local_swa_block_idxs.size()
                         << ", swa_block_byte_size=" << swa_ctx.swa_block_byte_size
                         << ", max_cached_len=" << swa_ctx.max_hit_token_num;
      }
    }
  }

  // decode 角色：QueryAndFetch 内部自动重试未命中 block，已拉取的不会重复拷贝
  // standalone / prefill 角色：单次查询
  KLLM_LOG_DEBUG << "QueryAndFetch, device_id=" << device_id_ << ", req_id=" << real_req->req_id
                 << ", token_num=" << real_req->token_ids.size() << ", local_cached_len=" << real_req->local_cached_len
                 << ", tp_size=" << tp_size << ", tp_rank=" << tp_rank << ", layer_index=" << 0
                 << ", block_size=" << real_req->block_size << ", block_byte_size=" << block_byte_size
                 << ", block_ptrs_size=" << block_ptrs.size() << ", retry=" << is_decode_role
                 << ", timeout_ms=" << timeout_ms_;
  const int64_t qf_start_us = ProfileTimer::GetCurrentTimeInUs();
  Status status =
      connector_->QueryAndFetch(real_req->token_ids, real_req->local_cached_len, tp_size, tp_rank,
                                /*layer_index=*/0, real_req->block_size, block_byte_size, block_ptrs, matched_ranges,
                                /*retry_until_full_hit=*/is_decode_role, timeout_ms_, use_swa ? &swa_ctx : nullptr,
                                use_swa ? &swa_result : nullptr);
  qf_us = ProfileTimer::GetCurrentTimeInUs() - qf_start_us;
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

    const int64_t token_start_us = ProfileTimer::GetCurrentTimeInUs();
    if (is_decode_role && all_blocks_hit) {
      // decode 角色：KV 已全命中，重试查询 token 直到找到或剩余时间耗尽
      const int64_t deadline_us = token_start_us + static_cast<int64_t>(timeout_ms_) * 1000;
      constexpr int kRetryIntervalMs = 10;

      while (ProfileTimer::GetCurrentTimeInUs() < deadline_us) {
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
    token_us = ProfileTimer::GetCurrentTimeInUs() - token_start_us;
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

void GlobalCacheQueryCallable::SetSwaBlockAllocatorManager(
    std::shared_ptr<BlockAllocatorManagerInterface> swa_block_allocator_manager) {
  swa_block_allocator_manager_ = swa_block_allocator_manager;
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
