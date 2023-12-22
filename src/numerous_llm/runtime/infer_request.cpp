/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/infer_request.h"

#include <atomic>
#include <limits>
#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/runtime/infer_stage.h"
#include "numerous_llm/utils/singleton.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/string_utils.h"
#include "numerous_llm/utils/tensor.h"

static std::atomic index_counter = 0;

namespace numerous_llm {

InferRequest::InferRequest() {
  constexpr int max = std::numeric_limits<int>::max();
  ++index_counter;
  if (index_counter == max) {
    index_counter = 1;
  }
  infer_id = index_counter;

  timestamp_in_ms = GetCurrentTimeInMs();
}

InferRequest::~InferRequest() {
  for (auto& blocks : kv_cache_blocks) {
    // Singleton<BlockManager>::GetInstance()->FreeBlocks(blocks);
  }
}

std::vector<float*> InferRequest::GetLogitsPtr() { return model_instance->GetLogitsPtr(); }

std::vector<std::vector<void*>> InferRequest::GetBlockPtrs() { return {{nullptr}}; }

size_t InferRequest::GetBlockSize() const { return 4096; }

void InferRequest::ResetInferStage() {
  NLLM_LOG_INFO << "input tokens number " << input_tokens.size();
  if (input_tokens.size() < output_tokens.size()) {
    NLLM_LOG_INFO << "change from context decode to decode";
    infer_stage = InferStage::STATE_DECODE;
  }
}

size_t InferRequest::GetStepTokenNumber() {
  size_t step_token_num = 1;
  if (infer_stage == STAGE_CONTEXT) {
    step_token_num += input_tokens.size();
  }
  return step_token_num;
}

size_t InferRequest::GetTotalTokenNumber() { return input_tokens.size() + output_tokens.size() + 1; }

size_t InferRequest::GetStepBlockNumber() {
  // size_t block_size = Singleton<BlockManager>::GetInstance()->GetBlockSize();
  // return ((model_instance->GetTokenCacheSize() * GetTotalTokenNumber() - 1)) / block_size + 1;
  return 0;
}

size_t InferRequest::GetTotalBlockNumber() {
  // size_t block_size = Singleton<BlockManager>::GetInstance()->GetBlockSize();
  // return ((model_instance->GetTokenCacheSize() * GetStepTokenNumber() - 1)) / block_size + 1;
  return 0;
}

Status InferRequest::SwapInAsync() {
  // Singleton<BlockManager>::GetInstance()->SwapIn(kv_cache_blocks[0], NULL);
  return Status();
}

Status InferRequest::SwapOutAsync() {
  // Singleton<BlockManager>::GetInstance()->SwapOut(kv_cache_blocks[0], NULL);
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

}  // namespace numerous_llm
