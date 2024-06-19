/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <stdint.h>

namespace llm_kernels {
namespace ascend {

struct EmbeddingConfigTiling {
  uint32_t total_seq_len = 0;
  uint32_t hidden_units = 0;
  uint32_t batch_size = 0;
  uint32_t tile_num = 1;
  size_t vocab_size = 0;
  // NOTE(karlluo): if using multiple *PU's emb, each *PU has one emb slice tagged by vocab_id.
  size_t vocab_id = 0;
  int32_t start_step = 0;
};

}  // namespace ascend
}  // namespace llm_kernels