/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>

namespace llm_kernels {
namespace ascend {

// Same as TCubeTiling, use this to avoid compile error.
struct PagedTCubeTiling {
  int32_t usedCoreNum;
  int32_t M;
  int32_t N;
  int32_t Ka;
  int32_t Kb;
  int32_t singleCoreM;
  int32_t singleCoreN;
  int32_t singleCoreK;
  int32_t baseM;
  int32_t baseN;
  int32_t baseK;
  int32_t depthA1;
  int32_t depthB1;
  int32_t stepM;
  int32_t stepN;
  int32_t isBias;
  int32_t transLength;
  int32_t iterateOrder;
  int32_t shareMode;
  int32_t shareL1Size;
  int32_t shareL0CSize;
  int32_t shareUbSize;
  int32_t batchM;
  int32_t batchN;
  int32_t singleBatchM;
  int32_t singleBatchN;
  int32_t stepKa;
  int32_t stepKb;
  int32_t dbL0A;
  int32_t dbL0B;
  int32_t dbL0C;
  int32_t ALayoutInfoB;
  int32_t ALayoutInfoS;
  int32_t ALayoutInfoN;
  int32_t ALayoutInfoG;
  int32_t ALayoutInfoD;
  int32_t BLayoutInfoB;
  int32_t BLayoutInfoS;
  int32_t BLayoutInfoN;
  int32_t BLayoutInfoG;
  int32_t BLayoutInfoD;
  int32_t CLayoutInfoB;
  int32_t CLayoutInfoS1;
  int32_t CLayoutInfoN;
  int32_t CLayoutInfoG;
  int32_t CLayoutInfoS2;
  int32_t BatchNum;
  int32_t reserved;
};

// Same as SoftMaxTiling, use this to avoid compile error.
struct PagedSoftMaxTiling {
  uint32_t srcM;
  uint32_t srcK;
  uint32_t srcSize;
  uint32_t outMaxM;
  uint32_t outMaxK;
  uint32_t outMaxSize;
  uint32_t splitM;
  uint32_t splitK;
  uint32_t splitSize;
  uint32_t reduceM;
  uint32_t reduceK;
  uint32_t reduceSize;
  uint32_t rangeM;
  uint32_t tailM;
  uint32_t tailSplitSize;
  uint32_t tailReduceSize;
};

// The tiling struct of paged attention, for one sequence.
struct PagedAttentionTilingData {
  // The cube tiling, used for prefill, do not change the field position.
  PagedTCubeTiling cube_tiling_qk;
  PagedTCubeTiling cube_tiling_wv;

  // The tiling struct for softmax, do not change field position.
  PagedSoftMaxTiling softmax_tiling;

  // The tiling key, used to specify data type.
  uint32_t data_type;

  // Whether it is a context decode stage, 0 or 1.
  uint32_t context_stage;

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
