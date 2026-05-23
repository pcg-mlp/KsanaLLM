/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "runtime/model_input.h"
#include "runtime/schedule_output.h"
#include "utils/status.h"

namespace ksana_llm {

class ModelBuilder {
 public:
  explicit ModelBuilder(std::shared_ptr<Context> context);

  Status BuildModelInput(std::shared_ptr<ScheduleOutput> schedule_output, std::shared_ptr<NewModelInput> model_input,
                         const RunMode run_mode);

  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager);

  void SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager);

  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);

  // Reorder the infer_request list, placing the requests from the Multi-Token Forwarding at the front
  // and the requests from the Single-Token Forwarding at the back.
  template <typename T>
  void ReorderInferRequests(std::vector<T> &reqs) {
    // Due to the different calculation logic used for multi-token and single-token in the Attention layer,
    // the requests are first sorted to utilize contiguous space for accelerated inference.
    std::sort(reqs.begin(), reqs.end(), [this](const auto &a, const auto &b) {
      // For dp case, the order is: [group1_prefill, group1_decode, group2_prefill, group2_decode, ...]
      // NOTE: prefill may appear after decode (if they are in different dp groups)
      if (a->attn_dp_group_id != b->attn_dp_group_id) {
        return a->attn_dp_group_id < b->attn_dp_group_id;
      }

      auto get_vec_size = [](const auto &item) {
        if constexpr (std::is_same_v<std::decay_t<decltype(item)>, std::vector<int>>) {
          return item.size();
        } else {
          return item->size();
        }
      };

      // For non-dp case, the order is: [prefill, decode]
      const size_t a_token_num = get_vec_size(a->forwarding_tokens) - a->kv_cached_token_num;
      const size_t b_token_num = get_vec_size(b->forwarding_tokens) - b->kv_cached_token_num;

      const bool is_a_decode = a_token_num <= GetDecodeTokenNumThreshold() && a->kv_cached_token_num > 0;
      const bool is_b_decode = b_token_num <= GetDecodeTokenNumThreshold() && b->kv_cached_token_num > 0;

      if (is_a_decode == is_b_decode) {
        // Both prefill or decode, sort the infer_reqs list based on a_token_num and b_token_num,
        // i.e., the number of tokens that need to be calculated for the KV cache.
        // NOTE: a_token_num or b_token_num may be zero
        if (a_token_num != b_token_num) {
          return a_token_num > b_token_num;
        }
        if (a->kv_cached_token_num != b->kv_cached_token_num) {
          return a->kv_cached_token_num < b->kv_cached_token_num;
        }
        return a->req_id < b->req_id;
      } else {
        // One is prefill, another is decode, prefill before decode
        return !is_a_decode;
      }
    });

    // reset logits offset after reorder
    size_t logits_offset = 0;
    for (auto &req : reqs) {
      req->logits_offset = logits_offset;
      logits_offset += req->sampling_token_num;
    }
  }

 private:
  // Build forward request, group by model name and stage.
  void BuildForwardRequests(std::vector<std::shared_ptr<ExecutorInferRequest>> &reqs,
                            std::vector<ForwardRequest> &forward_reqs);

  void BuildForwardRequestFromInferRequest(ForwardRequest &forward_req,
                                           std::shared_ptr<ExecutorInferRequest> &infer_req, uint32_t layer_num);

  // Build ATB KV cache block ids
  void BuildFlatKVCacheBlkIds(uint32_t layer_num, const std::vector<int> &device_block_ids, size_t rank_num,
                              std::vector<std::vector<int32_t>> &atb_block_ids,
                              std::shared_ptr<BlockAllocatorInterface> block_allocator);

 private:
  std::shared_ptr<Context> context_ = nullptr;

  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_ = nullptr;
  // The SWA block allocator manager; nullptr when SWA is disabled.
  std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager_ = nullptr;
};

}  // namespace ksana_llm
