/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"

namespace llm_kernels {
namespace ascend {

// The tiling struct of paged attention, for one sequence.
struct PagedAttentionTilingData {
  TCubeTiling cube_tiling_qk;
  TCubeTiling cube_tiling_wv;

  SoftMaxTiling softmax_tiling;

  // The tiling key, used to specify data type.
  uint32_t data_type;

  // Whether it is a context decode stage, 0 or 1.
  uint32_t multi_token_forward;

  // The seq len of current prompt. for decode stage, it is always 1.
  uint32_t seq_len;

  // The blocks num of current prompts.
  uint32_t seq_block_num;

  // The token num for every block.
  uint32_t block_token_num;

  // The max position of the last token.
  uint32_t token_pos;

  // The head size of model.
  uint32_t head_size;

  // The dimension of every head.
  uint32_t head_dim;

  // Used to construct attn mask.
  uint32_t max_seq_len;

  // The scale in 32 bit.
  uint32_t scale;
  uint32_t scale_fp16;

  // The used aicore block number.
  uint32_t used_core_num;

  // The ub tmp buffer size.
  uint32_t qk_ub_size;
  uint32_t wv_ub_size;
};

}  // namespace ascend
}  // namespace llm_kernels
