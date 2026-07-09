/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/model_builder.h"

#include "profiler/profile_event.h"
#include "runtime/fast_path_controller.h"
#include "runtime/infer_stage.h"
#include "utils/logger.h"
#include "utils/singleton.h"

namespace ksana_llm {

ModelBuilder::ModelBuilder(std::shared_ptr<Context> context) : context_(context) {}

void ModelBuilder::SetModelInstance(std::shared_ptr<ModelInstance> model_instance) { model_instance_ = model_instance; }

void ModelBuilder::SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager) {
  block_allocator_manager_ = block_allocator_manager;
}

void ModelBuilder::SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager) {
  swa_block_allocator_manager_ = swa_block_allocator_manager;
}

void ModelBuilder::BuildForwardRequests(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                                        std::vector<ForwardRequest>& forward_reqs) {
  PROFILE_EVENT_SCOPE(build_forward_requests, "Exec.BuildModelInput.BuildForwardRequests");
  for (auto& req_ptr : reqs) {
    req_ptr->block_allocator_manager = block_allocator_manager_;
    req_ptr->model_instance = model_instance_;
    req_ptr->step += 1;

    ForwardRequest& forward_req = forward_reqs.emplace_back();
    BuildForwardRequestFromInferRequest(forward_req, req_ptr, model_instance_->GetLayerNum());
  }
}

void ModelBuilder::BuildForwardRequestFromInferRequest(ForwardRequest& forward_req,
                                                       std::shared_ptr<ExecutorInferRequest>& req_ptr,
                                                       uint32_t layer_num) {
  const size_t attn_dp_group_id = context_->device->GetDeviceId() / context_->global->GetAttentionTensorParallelSize();
  const size_t attn_dp_rank_id = context_->device->GetDeviceId() % context_->global->GetAttentionTensorParallelSize();

  // Skip all kv blocks that not belong to current db group.
  if (attn_dp_group_id == req_ptr->attn_dp_group_id) {
    forward_req.kv_cache_ptrs = req_ptr->GetBlockPtrs(attn_dp_rank_id);

    // Populate SWA block base address when SWA is enabled.
    // Since SWA blocks are contiguous in memory (each block is block_size apart),
    // only the base address of block_id=0 is needed.
    if (swa_block_allocator_manager_ && !req_ptr->swa_kv_cache_blocks.empty()) {
      auto swa_group = swa_block_allocator_manager_->GetBlockAllocatorGroup(req_ptr->attn_dp_group_id);
      std::vector<void*> swa_block_ptrs(1);
      swa_group->GetDeviceBlockAllocator(attn_dp_rank_id)->GetBlockPtrs({0}, swa_block_ptrs);
      forward_req.swa_block_base_addr = swa_block_ptrs[0];
      forward_req.swa_kv_cache_blocks = req_ptr->swa_kv_cache_blocks;
      // SWA pool 总 block 数: attention kernel 需要它做 indices 上限校验。
      forward_req.swa_pool_block_num = static_cast<int64_t>(swa_group->GetBlockAllocatorGroupConfig().device_block_num);
    }
  }

  forward_req.req_id = req_ptr->req_id;
  forward_req.infer_stage = req_ptr->infer_stage;
  forward_req.step = req_ptr->step;
  forward_req.kv_cached_token_num = req_ptr->kv_cached_token_num;
  forward_req.kv_from_remote = req_ptr->kv_from_remote;
  // pd_v2 Prefill node: non-empty decode targets means this req must RDMA its
  // K/V to a Decode peer, so it must run the flash/context kernel (which holds
  // the RecordLayerProgress KV-dispatch hook) even for a 1-token full-hit.
  forward_req.kv_to_remote = !req_ptr->pd_v2_decode_targets.empty();
  forward_req.logits_custom_length = req_ptr->logits_custom_length;
  forward_req.sampling_token_num = req_ptr->sampling_token_num;
  forward_req.last_step_token_num = req_ptr->last_step_token_num;
  forward_req.logits_offset = req_ptr->logits_offset;
  forward_req.request_target = std::make_shared<const decltype(req_ptr->request_target)>(req_ptr->request_target);
  forward_req.response = &req_ptr->response;
  // forwarding_tokens 绑定策略 (与 build_ahead 开关联动):
  //   fast path: 深拷贝 — max_depth=2 时后续 schedule 会覆写 ExecutorInferRequest::forwarding_tokens,
  //              已入队待 Forward 的 ForwardRequest 须与 live 向量隔离.
  //   slow path: shared_ptr aliasing — 无并发 launch，无覆写风险。
  if (FastPathController::GetInstance().IsEnabledAtStartup()) {
    forward_req.forwarding_tokens = std::make_shared<std::vector<int>>(req_ptr->forwarding_tokens);
  } else {
    forward_req.forwarding_tokens = std::shared_ptr<std::vector<int>>(req_ptr, &req_ptr->forwarding_tokens);
  }
  forward_req.flexible_cached_copy_tasks = &(req_ptr->flexible_cached_copy_tasks);
  forward_req.input_refit_embedding = &(req_ptr->input_refit_embedding);
  forward_req.mrotary_embedding_pos_offset = &(req_ptr->mrotary_embedding_pos_offset);
  forward_req.xdrotary_embedding_pos_offset = &(req_ptr->xdrotary_embedding_pos_offset);
  forward_req.flexible_cache_len = req_ptr->flexible_cached_copy_tasks.size();
  forward_req.prefix_cache_len = req_ptr->prefix_cache_len + forward_req.flexible_cache_len;
  forward_req.sampling_config = &(req_ptr->sampling_config);
  forward_req.attn_dp_group_id = req_ptr->attn_dp_group_id;
  forward_req.block_allocator_manager = block_allocator_manager_;
  forward_req.swa_kv_cache_block_idx_offsets = req_ptr->swa_kv_cache_block_idx_offsets;

  // Skip all kv blocks that not belong to current db group.
  if (attn_dp_group_id == req_ptr->attn_dp_group_id) {
    auto block_allocator_group = block_allocator_manager_->GetBlockAllocatorGroup(forward_req.attn_dp_group_id);
    size_t rank_num = block_allocator_group->GetBlockAllocatorDevices().size();
    BuildFlatKVCacheBlkIds(layer_num, req_ptr->kv_cache_blocks, rank_num, forward_req.atb_kv_cache_base_blk_ids,
                           block_allocator_group->GetDeviceBlockAllocator(attn_dp_rank_id));
  }
}

// NOTE(karlluo): for ATB, all device blocks locate on a flatten plane memory space.
// The Ksana kv cache consists of blocks, each of which is an independent storage space. The blocks are not
// guaranteed to be contiguous in memory. Each block has a shape of [2, layer_num, block_token_num, head_num,
// head_dim], where 2 represents key and value. The Ascend ATB kv cache consists of kcache and vcache, which are
// independent contiguous storage spaces. The shapes of kcache and vcache are [num_blocks * layer_num,
// block_token_num, head_num, head_dim]. Each block has a size of [block_token_num, head_num, head_dim]. To
// interface with the NPU, Ascend ATB (hereinafter referred to as ATB) needs to be used. In order for the NPU's
// self/paged attention to utilize Ksana's kv cache and share the underlying memory/GPU memory management
// capabilities, the Ksana kv cache needs to be converted to the Ascend ATB kv cache format.
// 1. Change the block allocation method so that the blocks are contiguous in physical memory, while the upper-level
// pointers point to different storage spaces. Originally, each block in the Ksana kv cache called malloc once. This
// should be changed to pre-allocate a contiguous storage space of size [num_blocks, 2, layer_num, block_token_num,
// head_num, head_dim]. The pointers of each block should then point to cache_base_ptr + (block index * 2 *
// layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
// 2. During each inference process, each prompt will carry an array of block IDs, which can be used to obtain the
// pointers to the storage space. For ATB, conversion is required to use these pointers. The conversion process is
// as follows:
//    - Given a block ID array [b0, b1, b2, b3, b4] and the base address pointer of the Ksana kv cache after the
//    modification in step 1, cache_base_ptr.
//    - For ATB: The Ksana kv cache has a total of num_blocks * 2 * layer_num blocks.
//    - Therefore, the block ID array for ATB is [b0 * layer_num * 2, b1 * layer_num * 2, b2 * layer_num * 2, b3 *
//    layer_num * 2, b4 * layer_num * 2].
//    - Ksana's kv cache swaps memory/GPU memory at the block level, so to reuse Ksana's kv cache's underlying
//    memory/GPU memory management capabilities, ATB's kcache and vcache share the same Ksana kv cache.
//    - Since each block in Ksana is divided into K and V parts, each part having a size of [layer_num,
//    block_token_num, head_num, head_dim].
//    - To allow ATB's kcache and vcache to share the same block ID array, the kcache pointer is cache_base_ptr, and
//    the vcache pointer is cache_base_ptr + (layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
//    - Therefore, the block ID array for kcache/vcache is [b0 * layer_num * 2 + layer_idx, b1 * layer_num * 2 +
//    layer_idx, b2 * layer_num * 2 + layer_idx, b3 * layer_num * 2 + layer_idx, b4 * layer_num * 2 + layer_idx].
// More detail refer to docs/Technology/kvcache-relationship-between-ascend-atb-and-ksana.md
void ModelBuilder::BuildFlatKVCacheBlkIds(uint32_t layer_num, const std::vector<int>& device_block_ids, size_t rank_num,
                                          std::vector<std::vector<int32_t>>& atb_block_ids,
                                          std::shared_ptr<BlockAllocatorInterface> block_allocator) {
  atb_block_ids.resize(rank_num);
  const ModelConfig& model_config = model_instance_->GetModelConfig();

  // for dedicate device kv blocks
  size_t base_id = block_allocator->GetBlocksBaseId();
  std::vector<int32_t> base_block_ids;
  base_block_ids.reserve(device_block_ids.size());
  std::transform(device_block_ids.begin(), device_block_ids.end(), std::back_inserter(base_block_ids),
                 [base_id, layer_num, &model_config](int block_id) {
                   const size_t original_block_id = block_id - base_id;
                   if (model_config.use_dsa) {
                     return original_block_id;
                   } else if (model_config.use_mla) {
                     return original_block_id * layer_num;
                   } else {
                     return original_block_id * layer_num * 2;
                   }
                 });

  for (size_t rank = 0; rank < rank_num; ++rank) {
    atb_block_ids[rank] = base_block_ids;
  }
}

Status ModelBuilder::BuildModelInput(std::shared_ptr<ScheduleOutput> schedule_output,
                                     std::shared_ptr<NewModelInput> model_input, const RunMode run_mode) {
  PROFILE_EVENT_SCOPE(build_model_input, "Exec.BuildModelInput.Total");
  model_input->schedule_output = schedule_output;
  model_input->run_mode = run_mode;

  {
    PROFILE_EVENT_SCOPE(reorder_infer_requests, "Exec.BuildModelInput.Reorder");
    ReorderInferRequests(model_input->schedule_output->executor_running_reqs);
  }
  BuildForwardRequests(model_input->schedule_output->executor_running_reqs, model_input->reqs);

  return Status();
}

}  // namespace ksana_llm
