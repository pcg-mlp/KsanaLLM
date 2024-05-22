/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/infer_request.h"

#include <atomic>
#include <limits>
#include <vector>
#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

InferRequest::InferRequest(std::shared_ptr<Request>& request, int index)
    : req_id(request->req_ids[index]),
      model_name(request->model_name),
      input_tokens(request->input_tokens),
      subinput_pos(request->subinput_pos),
      subinput_embedding(request->subinput_embedding),
      output_tokens(std::get<0>(request->output_group[index])),
      logprobs(std::get<1>(request->output_group[index])),
      sampling_config(request->sampling_config),
      waiter(request->waiter),
      step_waiter(request->step_waiter),
      finished(request->finisheds[index]),
      finish_status(request->finish_status),
      output_mutex(request->output_mutex),
      padded_size(request->padded_size) {
  timestamp_in_ms = GetCurrentTimeInMs();
}

InferRequest::~InferRequest() {
  NLLM_LOG_DEBUG << "req " << req_id << " destroyed, free blocks if necessary.";
  if (!kv_cache_blocks.empty()) {
    FreeBlocks();
  }
}

Status InferRequest::FreeBlocks() {
  NLLM_LOG_DEBUG << "req " << req_id << " free blocks.";
  // Free memory on every device.
  for (size_t i = 0; i < kv_cache_blocks.size(); ++i) {
    GetBlockManager()->SetDeviceId(i);
    if (is_use_prefix_cache) {
      NLLM_LOG_DEBUG << fmt::format(
          "req {} kv_cache_blocks[{}] len: {} include prefix share cache tokens number: {} in {} blocks", req_id, i,
          kv_cache_blocks[i].size(), prefix_cache_len, prefix_cache_blocks_number);
      // NOTE(karlluo): skip prefix share blocks
      std::vector<int> private_kv_cache_blocks(kv_cache_blocks[i].size() - prefix_cache_blocks_number);
      std::copy(kv_cache_blocks[i].begin() + prefix_cache_blocks_number, kv_cache_blocks[i].end(),
                private_kv_cache_blocks.begin());
      GetBlockManager()->FreeBlocks(private_kv_cache_blocks);
    } else {
      GetBlockManager()->FreeBlocks(kv_cache_blocks[i]);
    }
  }
  kv_cache_blocks.clear();
  return Status();
}

void InferRequest::Notify() {
  for (int i = 0; i < req_group.size(); i++) {
    if (!req_group[i]->finished) return;
  }
  for (int i = 0; i < req_group.size(); i++) {
    req_group[i]->ClearReqGroup();
  }
  if (waiter) {
    waiter->Notify();
  }
  if (step_waiter) {
    step_waiter->Notify();
  }
}

void InferRequest::NotifyStep() {
  if (sampling_config.num_beams > 1) {
    int output_tokens_len = -1;
    for (int i = 0; i < req_group.size(); i++) {
      if (req_group[i]->finished) continue;
      output_tokens_len = output_tokens_len == -1 ? req_group[i]->output_tokens.size() : output_tokens_len;
      if (req_group[i]->output_tokens.size() != output_tokens_len) return;
    }
  }

  if (step_waiter) {
    step_waiter->Notify();
  }
}

std::vector<float*> InferRequest::GetLogitsPtr() { return model_instance->GetLogitsPtr(); }

std::vector<std::vector<void*>> InferRequest::GetBlockPtrs() {
  std::vector<std::vector<void*>> block_ptrs;
  for (int rank = 0; rank < kv_cache_blocks.size(); ++rank) {
    std::vector<void*> block_ptr(kv_cache_blocks[rank].size());
    GetBlockManager()->SetDeviceId(rank);
    GetBlockManager()->GetBlockPtrs(kv_cache_blocks[rank], block_ptr);
    block_ptrs.push_back(block_ptr);
  }
  return block_ptrs;
}

void InferRequest::AdjustInferStage() {
  if (input_tokens.size() < output_tokens.size()) {
    infer_stage = InferStage::STATE_DECODE;
  }
}

size_t InferRequest::GetStepTokenNumber() {
  size_t step_token_num = 1;
  if (infer_stage == STAGE_CONTEXT) {
    step_token_num += output_tokens.size();
  }
  return step_token_num;
}

size_t InferRequest::GetTotalTokenNumber() {
  if (is_use_prefix_cache) {
    return output_tokens.size() + 1 - prefix_cache_len;
  } else {
    return output_tokens.size() + 1;
  }
}

size_t InferRequest::GetStepBlockNumber() {
  size_t block_token_num = GetBlockManager()->GetBlockTokenNum();
  size_t last_token_num = GetTotalTokenNumber() - GetStepTokenNumber();
  return GetTotalBlockNumber() - ((block_token_num + last_token_num - 1) / block_token_num);
}

size_t InferRequest::GetTotalBlockNumber() {
  size_t block_token_num = GetBlockManager()->GetBlockTokenNum();
  return (block_token_num + GetTotalTokenNumber() - 1) / block_token_num;
}

size_t InferRequest::GetCurrentBlockNumber() {
  if (!kv_cache_blocks.empty()) {
    return kv_cache_blocks[0].size() - prefix_cache_blocks_number;
  }
  return 0;
}

Status InferRequest::SwapInAsync() {
  for (size_t i = 0; i < kv_cache_blocks.size(); ++i) {
    std::vector<int> device_blocks;
    GetBlockManager()->SetDeviceId(i);
    if (is_use_prefix_cache) {
      NLLM_LOG_DEBUG << fmt::format(
          "req {} kv_cache_blocks[{}] len: {} include prefix share cache tokens number: {} in {} blocks", req_id, i,
          kv_cache_blocks[i].size(), prefix_cache_len, prefix_cache_blocks_number);
      // NOTE(karlluo): skip prefix share blocks
      std::vector<int> private_kv_cache_blocks(kv_cache_blocks[i].size() - prefix_cache_blocks_number);
      std::copy(kv_cache_blocks[i].begin() + prefix_cache_blocks_number, kv_cache_blocks[i].end(),
                private_kv_cache_blocks.begin());
      GetBlockManager()->SwapIn(private_kv_cache_blocks, device_blocks);
      std::copy(device_blocks.begin(), device_blocks.end(), kv_cache_blocks[i].begin() + prefix_cache_blocks_number);
    } else {
      NLLM_LOG_DEBUG << fmt::format("req {} kv_cache_blocks[{}] len: {}", req_id, i, kv_cache_blocks[i].size());
      GetBlockManager()->SwapIn(kv_cache_blocks[i], device_blocks);
      kv_cache_blocks[i].swap(device_blocks);
    }
  }

  return Status();
}

Status InferRequest::SwapOutAsync(const int host_block_num_to_add) {
  for (size_t i = 0; i < kv_cache_blocks.size(); ++i) {
    std::vector<int> host_blocks;
    GetBlockManager()->SetDeviceId(i);
    if (is_use_prefix_cache) {
      NLLM_LOG_DEBUG << fmt::format(
          "req {} kv_cache_blocks[{}] len: {} include prefix share cache tokens number: {} in {} blocks", req_id, i,
          kv_cache_blocks[i].size(), prefix_cache_len, prefix_cache_blocks_number);
      // NOTE(karlluo): skip prefix share blocks
      std::vector<int> private_kv_cache_blocks(kv_cache_blocks[i].size() - prefix_cache_blocks_number);
      std::copy(kv_cache_blocks[i].begin() + prefix_cache_blocks_number, kv_cache_blocks[i].end(),
                private_kv_cache_blocks.begin());
      GetBlockManager()->SwapOut(private_kv_cache_blocks, host_blocks, host_block_num_to_add);
      std::copy(host_blocks.begin(), host_blocks.end() - host_block_num_to_add, kv_cache_blocks[i].begin() + prefix_cache_blocks_number);
    } else {
      GetBlockManager()->SwapOut(kv_cache_blocks[i], host_blocks, host_block_num_to_add);
      kv_cache_blocks[i].swap(host_blocks);
    }
  }

  return Status();
}

Status InferRequest::DropSwappedAsync() {
  for (size_t i = 0; i < kv_cache_blocks.size(); ++i) {
    GetBlockManager()->SetDeviceId(i);
    if (is_use_prefix_cache) {
      // NOTE(karlluo): skip prefix share blocks
      std::vector<int> private_kv_cache_blocks(kv_cache_blocks[i].size() - prefix_cache_blocks_number);
      std::copy(kv_cache_blocks[i].begin() + prefix_cache_blocks_number, kv_cache_blocks[i].end(),
                private_kv_cache_blocks.begin());
      GetBlockManager()->SwapDrop(private_kv_cache_blocks);
    } else {
      GetBlockManager()->SwapDrop(kv_cache_blocks[i]);
    }
  }

  return Status();
}

bool InferRequest::CheckLoraEnable() {
  // TODO
  return false;
}

size_t InferRequest::GetLoraBlockNumber() {
  // TODO
  return 0;
}

Status InferRequest::SwapInLoraAsync() {
  // TODO
  return Status();
}

Status InferRequest::SwapOutLoraAsync() {
  // TODO
  return Status();
}

Status InferRequest::AllocateStepBlocks() {
  // TODO
  return Status();
}

}  // namespace ksana_llm
