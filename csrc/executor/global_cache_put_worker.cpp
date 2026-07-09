/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/global_cache_put_worker.h"

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
                                           std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper, const std::string& role)
    : connector_(connector),
      block_allocator_manager_(block_allocator_manager),
      dp_group_id_(dp_group_id),
      device_id_(device_id),
      ipc_func_wrapper_(ipc_func_wrapper),
      role_(role) {}

GlobalCachePutWorker::~GlobalCachePutWorker() { Stop(); }

void GlobalCachePutWorker::Start() {
  thread_pool_ = std::make_unique<ThreadPool>(2);
  thread_pool_->Start();
  KLLM_LOG_INFO << "started, device_id=" << device_id_;
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
    // 快照 forwarding_tokens，避免异步执行时被调度器修改（standalone 模式下
    // sampling 完成后调度器会向 forwarding_tokens 追加新生成的 token，导致
    // hash 与 QueryAndFetch 侧不一致）
    auto token_snapshot = std::make_shared<std::vector<int>>(req->forwarding_tokens);
    auto req_copy = req;
    thread_pool_->Submit([this, req_copy, token_snapshot, ipc, device_id]() {
      SetDevice(device_id);
      PutRequestKVCache(req_copy, *token_snapshot);
      if (ipc) {
        Status status = ipc->ReportGlobalCachePutDone(device_id, req_copy->req_id);
        if (!status.OK()) {
          KLLM_LOG_ERROR << "report put done failed, req_id=" << req_copy->req_id << ", device_id=" << device_id;
        }
      }
    });
  }
}

void GlobalCachePutWorker::PutRequestKVCache(const std::shared_ptr<ExecutorInferRequest>& req,
                                             const std::vector<int>& token_ids) {
  if (!connector_ || !block_allocator_manager_) {
    KLLM_LOG_WARNING << "connector or block_allocator_manager not set";
    return;
  }

  // 获取 block 对应的 GPU 内存指针
  auto block_allocator_group = block_allocator_manager_->GetBlockAllocatorGroup(dp_group_id_);
  if (!block_allocator_group) {
    KLLM_LOG_WARNING << "block_allocator_group is null, dp_group_id=" << dp_group_id_;
    return;
  }

  size_t block_byte_size = block_allocator_group->GetBlockAllocatorGroupConfig().block_size;

  auto device_allocator = block_allocator_group->GetDeviceBlockAllocator(device_id_);
  if (!device_allocator) {
    KLLM_LOG_WARNING << "device_allocator is null, device_id=" << device_id_;
    return;
  }

  std::vector<void*> block_ptrs;
  device_allocator->GetBlockPtrs(req->kv_cache_blocks, block_ptrs);

  if (block_ptrs.empty()) {
    KLLM_LOG_WARNING << "block_ptrs is empty, req_id=" << req->req_id;
    return;
  }

  // 使用提交时快照的 token_ids，而非 req->forwarding_tokens（后者可能已被调度器修改）
  size_t total_tokens = token_ids.size();
  if (total_tokens == 0) {
    return;
  }

  std::vector<int> block_ranges = {0, static_cast<int>(total_tokens - 1)};

  int tp_size = 1;
  int tp_rank = 0;

  KLLM_LOG_DEBUG << "PutGlobalCacheBlocks, device_id=" << device_id_ << ", req_id=" << req->req_id
                 << ", token_num=" << total_tokens << ", block_size=" << req->block_token_num
                 << ", block_byte_size=" << block_byte_size << ", block_ptrs_size=" << block_ptrs.size()
                 << ", block_ranges=" << block_ranges;
  Status status = connector_->PutGlobalCacheBlocks(token_ids, tp_size, tp_rank,
                                                   /*layer_index=*/0, static_cast<int>(req->block_token_num),
                                                   block_byte_size, block_ranges, block_ptrs);

  if (!status.OK()) {
    KLLM_LOG_WARNING << "PutGlobalCacheBlocks failed, req_id=" << req->req_id << ", error=" << status.GetMessage();
  } else {
    KLLM_LOG_SCHEDULER << "KV cache put done, req_id=" << req->req_id << ", tokens=" << total_tokens
                       << ", device_id=" << device_id_;
  }
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

    // 合并 generated_tokens + draft_tokens（MTP 产生的推测 token）
    std::vector<int> tokens_to_save;
    tokens_to_save.reserve(req->generated_tokens.size() + req->draft_tokens.GetDraftTokens().size());
    tokens_to_save.insert(tokens_to_save.end(), req->generated_tokens.begin(), req->generated_tokens.end());
    auto drafts = req->draft_tokens.GetDraftTokens();
    tokens_to_save.insert(tokens_to_save.end(), drafts.begin(), drafts.end());

    auto token_ids = req->input_tokens;

    KLLM_LOG_DEBUG << "SubmitFirstTokensPut: req_id=" << req->req_id
                   << ", generated_count=" << req->generated_tokens.size() << ", draft_count=" << drafts.size()
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
