/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>

#include "ksana_llm/runtime/infer_stage.h"

namespace ksana_llm {

// The information used for forward.
struct ForwardRequest {
  // The infer stage, context decode or decode.
  InferStage infer_stage;

  // The decode step, 1 for context decode, and then 2, 3, 4...
  int step;

  // The input tokens.
  std::vector<int>* output_tokens;

  // The output logits buf and offset.
  std::vector<float*> logits_buf;
  size_t logits_offset;

  // The kv cache addresses, for every device.
  std::vector<std::vector<void*>> kv_cache_ptrs;

  // The block size for every kv cache block.
  size_t block_size;
};

}  // namespace ksana_llm
