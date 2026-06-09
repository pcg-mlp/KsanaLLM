/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/executor_infer_request.h"

#include <atomic>
#include <limits>
#include <sstream>
#include <vector>
#include "runtime/infer_stage.h"
#include "utils/memory_utils.h"
#include "utils/request.h"
#include "utils/singleton.h"
#include "utils/status.h"
#include "utils/string_utils.h"
#include "utils/tensor.h"

namespace ksana_llm {

ExecutorInferRequest::~ExecutorInferRequest() { KLLM_LOG_DEBUG << "req " << req_id << " destroyed."; }

std::string ExecutorInferRequest::PrintKVBlockIds(bool print_details) const {
  std::ostringstream ss;
  ss << ", kv_cache_blocks_size:" << kv_cache_blocks.size() << ", kv_cache_blocks: {";
  if (print_details) {
    ss << "{ ";
    for (auto blk_id : kv_cache_blocks) {
      ss << blk_id << ", ";
    }
    ss << "}, ";
  }
  ss << "}";
  return ss.str();
}

std::string ExecutorInferRequest::ToString(bool print_details) const {
  std::ostringstream oss;
  oss << " req(req_id:" << req_id << ", step:" << step << ", sampling_token_num:" << sampling_token_num
      << ", kv_cached_token_num:" << kv_cached_token_num << ", prefix_cache_len:" << prefix_cache_len
      << ", input_tokens_size:" << input_tokens.size() << ", output_tokens_size:" << output_tokens.size()
      << ", forwarding_tokens_size:" << forwarding_tokens.size() << ", draft_tokens_size:" << draft_tokens.size()
      << ", accepted_tokens_size:" << accepted_tokens.size() << ", generated_token_size:" << generated_tokens.size()
      << PrintKVBlockIds(print_details) << ", finished:" << finished << ", aborted:" << aborted
      << ", finish_status:" << finish_status.ToString() << " ) ";
  return oss.str();
}

std::ostream &operator<<(std::ostream &os, const ExecutorInferRequest &req) {
  os << req.ToString();
  return os;
}

void ExecutorInferRequest::Notify() {
  for (size_t i = 0; i < req_group.size(); i++) {
    if (!req_group[i]->finished) return;
  }

  if (sampling_config.num_beams > 1) {
    std::sort(beam_search_group.begin(), beam_search_group.end(),
              [](const OutputTuple &a, const OutputTuple &b) { return std::get<2>(a) > std::get<2>(b); });

    for (size_t i = 0; i < req_group.size() && i < beam_search_group.size(); i++) {
      req_group[i]->output_tokens = std::move(std::get<0>(beam_search_group[i]));
      req_group[i]->logprobs = std::move(std::get<1>(beam_search_group[i]));
    }
  }

  for (size_t i = 0; i < req_group.size(); i++) {
    req_group[i]->ClearReqGroup();
  }

  // After a notification, the corresponding request may be destructed.
  // So we return early to avoid accessing any variables referencing it.
  if (aborted) {
    abort_waiter->Notify();
    return;
  }
  if (waiter) {
    waiter->Notify();
    return;
  }
  if (step_waiter) {
    step_waiter->Notify();
  }
}

void ExecutorInferRequest::NotifyStep() {
  if (sampling_config.num_beams > 1) {
    int output_tokens_len = -1;
    for (size_t i = 0; i < req_group.size(); i++) {
      if (req_group[i]->finished) continue;
      output_tokens_len = output_tokens_len == -1 ? req_group[i]->output_tokens.size() : output_tokens_len;
      if (req_group[i]->output_tokens.size() != (size_t)output_tokens_len) return;
    }
  }

  if (step_waiter) {
    step_waiter->Notify();
  }
}

std::vector<int> ExecutorInferRequest::GetVerifiedTokens() {
  std::vector<int> tokens = forwarding_tokens;
  tokens.resize(forwarding_tokens.size() - forwarding_tokens_draft_num);
  tokens.insert(tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
  tokens.insert(tokens.end(), generated_tokens.begin(), generated_tokens.end());
  return tokens;
}

std::vector<void *> ExecutorInferRequest::GetBlockPtrs(int attn_dp_rank_id) {
  std::vector<void *> block_ptrs(kv_cache_blocks.size());
  block_allocator_manager->GetBlockAllocatorGroup(attn_dp_group_id)
      ->GetDeviceBlockAllocator(attn_dp_rank_id)
      ->GetBlockPtrs(kv_cache_blocks, block_ptrs);

  return block_ptrs;
}

std::vector<int> ExecutorInferRequest::GetKVOccupiedDevices() {
  std::vector<int> kv_occupied_devices;
  return kv_occupied_devices;
}

}  // namespace ksana_llm
