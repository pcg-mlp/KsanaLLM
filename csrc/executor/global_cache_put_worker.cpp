/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/global_cache_put_worker.h"

#include <algorithm>
#include <unordered_set>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "device/device_utils.h"
#include "global_cache_connector/global_cache_connector_interface.h"
#include "ipc/ipc_func_wrapper.h"
#include "runtime/executor_infer_request.h"
#include "utils/logger.h"

namespace ksana_llm {

GlobalCachePutWorker::GlobalCachePutWorker(std::shared_ptr<GlobalCacheConnectorInterface> connector,
                                           std::shared_ptr<BlockAllocatorManager> block_allocator_manager,
                                           int dp_group_id, int device_id,
                                           std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper, const std::string& role,
                                           int fetch_thread_num)
    : connector_(connector),
      block_allocator_manager_(block_allocator_manager),
      dp_group_id_(dp_group_id),
      device_id_(device_id),
      ipc_func_wrapper_(ipc_func_wrapper),
      role_(role),
      fetch_thread_num_(std::max(1, fetch_thread_num)) {}

GlobalCachePutWorker::~GlobalCachePutWorker() { Stop(); }

void GlobalCachePutWorker::SetSwaConfig(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager,
                                        int swa_tail_block_num, int swa_front_equal_parts) {
  swa_block_allocator_manager_ = swa_block_allocator_manager;
  swa_tail_block_num_ = std::max(1, swa_tail_block_num);
  // 至少 3 份才有「去掉首尾后仍可上传」的中间段；更小值按 3 钳制。
  swa_front_equal_parts_ = std::max(3, swa_front_equal_parts);
  KLLM_LOG_INFO << "swa put enabled, device_id=" << device_id_ << ", tail_block_num=" << swa_tail_block_num_
                << ", front_equal_parts=" << swa_front_equal_parts_
                << ", front_upload_num=" << (swa_front_equal_parts_ - 2);
}

void GlobalCachePutWorker::Start() {
  thread_pool_ = std::make_unique<ThreadPool>(static_cast<size_t>(fetch_thread_num_));
  thread_pool_->Start();
  KLLM_LOG_INFO << "started, device_id=" << device_id_ << ", fetch_thread_num=" << fetch_thread_num_;
}

void GlobalCachePutWorker::Stop() {
  terminated_.store(true, std::memory_order_relaxed);
  if (thread_pool_) {
    thread_pool_->Stop();
    thread_pool_.reset();
  }
  KLLM_LOG_INFO << "stopped, device_id=" << device_id_;
}

void GlobalCachePutWorker::SubmitBatchPut(const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  if (terminated_.load(std::memory_order_relaxed) || !connector_) {
    return;
  }

  auto ipc = ipc_func_wrapper_;
  int device_id = device_id_;

  for (const auto& req : reqs) {
    if (req->infer_stage != InferStage::kContext) {
      continue;
    }
    // 仅 put 已覆盖完整 input 的最后一段 prefill；中间 split-fuse chunk 的 forwarding 短于 input。
    const size_t draft_num = req->spec_state.forwarding_tokens_draft_num;
    const size_t covered = req->forwarding_tokens.size() >= draft_num ? req->forwarding_tokens.size() - draft_num : 0;
    if (covered < req->input_tokens.size()) {
      KLLM_LOG_DEBUG << "skip put, incomplete input, req_id=" << req->req_id << ", covered=" << covered
                     << ", input_tokens=" << req->input_tokens.size();
      continue;
    }
    // 快照 forwarding_tokens，避免异步执行时被调度器修改（standalone 模式下
    // sampling 完成后调度器会向 forwarding_tokens 追加新生成的 token，导致
    // hash 与 QueryAndFetch 侧不一致）
    auto token_snapshot = std::make_shared<std::vector<int>>(req->forwarding_tokens);
    // SWA block 列表同理：engine 在本步 forward 之后就会把滑出窗口的 block 从请求上摘掉，
    // 并通过下一步的 patch 覆写这两个数组。异步 put 真正执行时读到的会是滑动后的残余
    // （DSv4 只剩尾部那个不满块），必须在提交时刻定格。
    auto swa_snapshot = std::make_shared<SwaBlockSnapshot>();
    swa_snapshot->block_ids = req->swa_kv_cache_blocks;
    swa_snapshot->block_idx_offsets = req->swa_kv_cache_block_idx_offsets;
    auto req_copy = req;
    thread_pool_->Submit([this, req_copy, token_snapshot, swa_snapshot, ipc, device_id]() {
      SetDevice(device_id);
      const int fa_put_blocks = PutRequestKVCache(req_copy, *token_snapshot);
      const int swa_put_blocks = PutRequestSwaCache(req_copy, *token_snapshot, *swa_snapshot);
      if (ipc) {
        Status status = ipc->ReportGlobalCachePutDone(device_id, req_copy->req_id, fa_put_blocks, swa_put_blocks);
        if (!status.OK()) {
          KLLM_LOG_ERROR << "report put done failed, req_id=" << req_copy->req_id << ", device_id=" << device_id;
        }
      }
    });
  }
}

int GlobalCachePutWorker::PutRequestKVCache(const std::shared_ptr<ExecutorInferRequest>& req,
                                            const std::vector<int>& token_ids) {
  if (!connector_ || !block_allocator_manager_) {
    KLLM_LOG_WARNING << "connector or block_allocator_manager not set";
    return 0;
  }

  // 获取 block 对应的 GPU 内存指针
  auto block_allocator_group = block_allocator_manager_->GetBlockAllocatorGroup(dp_group_id_);
  if (!block_allocator_group) {
    KLLM_LOG_WARNING << "block_allocator_group is null, dp_group_id=" << dp_group_id_;
    return 0;
  }

  size_t block_byte_size = block_allocator_group->GetBlockAllocatorGroupConfig().block_size;

  auto device_allocator = block_allocator_group->GetDeviceBlockAllocator(device_id_);
  if (!device_allocator) {
    KLLM_LOG_WARNING << "device_allocator is null, device_id=" << device_id_;
    return 0;
  }

  std::vector<void*> block_ptrs;
  device_allocator->GetBlockPtrs(req->kv_cache_blocks, block_ptrs);

  if (block_ptrs.empty()) {
    KLLM_LOG_WARNING << "block_ptrs is empty, req_id=" << req->req_id;
    return 0;
  }

  // 使用提交时快照的 token_ids，而非 req->forwarding_tokens（后者可能已被调度器修改）
  size_t total_tokens = token_ids.size();
  if (total_tokens == 0) {
    return 0;
  }

  // SWA 开启时主 KV 也只存满 block：不满块前缀哈希依赖实际长度，存了也无法跨请求命中。
  size_t put_token_end = total_tokens;
  if (swa_block_allocator_manager_ != nullptr && req->block_token_num > 0) {
    put_token_end = (total_tokens / req->block_token_num) * req->block_token_num;
    if (put_token_end == 0) {
      return 0;
    }
  }
  std::vector<int> block_ranges = {0, static_cast<int>(put_token_end - 1)};

  int tp_size = 1;
  int tp_rank = 0;

  KLLM_LOG_DEBUG << "PutGlobalCacheBlocks, device_id=" << device_id_ << ", req_id=" << req->req_id
                 << ", token_num=" << total_tokens << ", put_token_end=" << put_token_end
                 << ", block_size=" << req->block_token_num << ", block_byte_size=" << block_byte_size
                 << ", block_ptrs_size=" << block_ptrs.size() << ", block_ranges=" << block_ranges;
  int put_ok = 0;
  Status status = connector_->PutGlobalCacheBlocks(token_ids, tp_size, tp_rank,
                                                   /*layer_index=*/0, static_cast<int>(req->block_token_num),
                                                   block_byte_size, block_ranges, block_ptrs, &put_ok);

  if (!status.OK()) {
    KLLM_LOG_WARNING << "PutGlobalCacheBlocks failed, req_id=" << req->req_id << ", error=" << status.GetMessage();
    return 0;
  }
  KLLM_LOG_SCHEDULER << "KV cache put done, req_id=" << req->req_id << ", tokens=" << total_tokens
                     << ", put_token_end=" << put_token_end << ", put_ok=" << put_ok << ", device_id=" << device_id_;
  return put_ok;
}

int GlobalCachePutWorker::PutRequestSwaCache(const std::shared_ptr<ExecutorInferRequest>& req,
                                             const std::vector<int>& token_ids, const SwaBlockSnapshot& swa_snapshot) {
  if (!connector_ || !swa_block_allocator_manager_) {
    return 0;
  }
  if (swa_snapshot.block_ids.empty() || req->block_token_num == 0) {
    return 0;
  }
  if (swa_snapshot.block_ids.size() != swa_snapshot.block_idx_offsets.size()) {
    KLLM_LOG_WARNING << "swa blocks/offsets size mismatch, req_id=" << req->req_id
                     << ", blocks=" << swa_snapshot.block_ids.size()
                     << ", offsets=" << swa_snapshot.block_idx_offsets.size();
    return 0;
  }

  const size_t total_tokens = token_ids.size();
  if (total_tokens == 0) {
    return 0;
  }

  auto swa_group = swa_block_allocator_manager_->GetBlockAllocatorGroup(dp_group_id_);
  if (!swa_group) {
    KLLM_LOG_WARNING << "swa block_allocator_group is null, dp_group_id=" << dp_group_id_;
    return 0;
  }
  const size_t swa_block_byte_size = swa_group->GetBlockAllocatorGroupConfig().block_size;
  auto swa_allocator = swa_group->GetDeviceBlockAllocator(device_id_);
  if (!swa_allocator || swa_block_byte_size == 0) {
    KLLM_LOG_WARNING << "swa allocator unavailable, device_id=" << device_id_
                     << ", swa_block_byte_size=" << swa_block_byte_size;
    return 0;
  }

  // 只存满块：不满块的前缀哈希取实际 token 数，不同长度的请求算出的 key 不同，存了也命中不了。
  const size_t full_block_num = total_tokens / req->block_token_num;
  if (full_block_num == 0) {
    return 0;
  }
  const size_t tail_start =
      full_block_num > static_cast<size_t>(swa_tail_block_num_) ? full_block_num - swa_tail_block_num_ : 0;

  // 目标集合 = 尾部 [tail_start, B) ∪ 前段等分后去掉首尾的中间份中位点，再与可见 offsets 求交。
  std::unordered_set<size_t> want_idxs;
  for (size_t idx = tail_start; idx < full_block_num; ++idx) {
    want_idxs.insert(idx);
  }
  const size_t front_len = tail_start;
  // 首尾两段不上传 → 中间 upload_n = parts - 2。
  const int parts = swa_front_equal_parts_;
  const int upload_n = parts - 2;
  if (front_len > 0 && upload_n > 0) {
    const int part_offset = 1;  // 跳过第 0 段，从第 1 段起取中间 upload_n 份
    for (int j = 0; j < upload_n; ++j) {
      const int part_i = part_offset + j;
      // 第 part_i 段中点：((2*i+1)/(2*parts)) * front_len
      size_t mid = (static_cast<size_t>(2 * part_i + 1) * front_len) / static_cast<size_t>(2 * parts);
      if (mid >= front_len) {
        mid = front_len - 1;
      }
      want_idxs.insert(mid);
    }
  }

  std::vector<size_t> selected_idxs;
  std::vector<int> selected_blocks;
  for (size_t i = 0; i < swa_snapshot.block_idx_offsets.size(); ++i) {
    const size_t idx = swa_snapshot.block_idx_offsets[i];
    if (idx >= full_block_num) {
      continue;
    }
    if (want_idxs.count(idx) != 0) {
      selected_idxs.push_back(idx);
      selected_blocks.push_back(swa_snapshot.block_ids[i]);
    }
  }

  if (selected_idxs.empty()) {
    KLLM_LOG_DEBUG << "no swa block selected, req_id=" << req->req_id << ", full_block_num=" << full_block_num
                   << ", visible=" << swa_snapshot.block_idx_offsets.size();
    return 0;
  }

  std::vector<void*> swa_block_ptrs;
  swa_allocator->GetBlockPtrs(selected_blocks, swa_block_ptrs);
  if (swa_block_ptrs.size() != selected_idxs.size()) {
    KLLM_LOG_WARNING << "swa GetBlockPtrs size mismatch, req_id=" << req->req_id
                     << ", expected=" << selected_idxs.size() << ", got=" << swa_block_ptrs.size();
    return 0;
  }

  // SWA 各卡内容不同，key 用本卡 device_id 作 tp_rank；主 KV 仍走 tp_rank=0 分片上传。
  int put_ok = 0;
  Status status = connector_->PutSwaBlocks(token_ids, /*tp_rank=*/device_id_, /*layer_index=*/0,
                                           static_cast<int>(req->block_token_num), swa_block_byte_size, selected_idxs,
                                           swa_block_ptrs, &put_ok);
  if (!status.OK()) {
    KLLM_LOG_WARNING << "PutSwaBlocks failed, req_id=" << req->req_id << ", error=" << status.GetMessage();
    return 0;
  }
  KLLM_LOG_SCHEDULER << "SWA cache put done, req_id=" << req->req_id << ", selected=" << selected_idxs.size() << "/"
                     << full_block_num << ", put_ok=" << put_ok << ", device_id=" << device_id_;
  return put_ok;
}

void GlobalCachePutWorker::SubmitFirstTokensPut(const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  if (terminated_.load(std::memory_order_relaxed) || !connector_) {
    return;
  }

  // 仅 prefill / pd_v2_prefill 角色 + 卡 0 保存; standalone (未设置 role) 和 decode 角色不保存
  if ((role_ != "prefill" && role_ != "pd_v2_prefill") || device_id_ != 0) {
    KLLM_LOG_DEBUG << "SubmitFirstTokensPut skipped: role=" << (role_.empty() ? "standalone" : role_)
                   << ", device_id=" << device_id_;
    return;
  }

  for (const auto& req : reqs) {
    if (req->generated_tokens.empty()) {
      continue;
    }

    // Merge generated_tokens + draft_tokens for cache persistence.
    auto tokens_to_save = GetAllProducedTokens(*req);

    auto token_ids = req->input_tokens;

    KLLM_LOG_DEBUG << "SubmitFirstTokensPut: req_id=" << req->req_id
                   << ", generated_count=" << req->generated_tokens.size()
                   << ", draft_count=" << req->spec_state.draft_tokens.size()
                   << ", total_save=" << tokens_to_save.size() << ", input_token_count=" << token_ids.size()
                   << ", first_token=" << tokens_to_save[0];

    int dev_id = device_id_;
    thread_pool_->Submit([this, token_ids = std::move(token_ids), tokens_to_save = std::move(tokens_to_save),
                          req_id = req->req_id, dev_id]() {
      SetDevice(dev_id);
      Status status = connector_->PutFirstTokens(token_ids, tokens_to_save);
      if (!status.OK()) {
        KLLM_LOG_WARNING << "PutFirstTokens failed, req_id=" << req_id << ", error=" << status.GetMessage();
      } else {
        KLLM_LOG_DEBUG << "PutFirstTokens done, req_id=" << req_id << ", token_count=" << tokens_to_save.size();
      }
    });
  }
}

}  // namespace ksana_llm
