/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/base/attention_meta/dsa_shared_attention_meta.h"

namespace ksana_llm {

// Compute the per-token KV cache write location (store_out_loc) for the DSA path.
// For each forwarding token, calculates the absolute slot index within its KV cache block
// and uploads the result to the device.
//
// Caller must ensure config.model_config.use_dsa is true before calling.
void PrepareStoreOutLoc(DSAAttentionMeta& dsa_meta, DSASharedAttentionMeta& dsa_shared, const AttentionMeta& input,
                        const AttentionBuilderConfig& config);

// Build and upload expanded_input_length for the DSA MTP decode path.
// Only active when input.q_seq_len > kDeepGemmMaxQSeqLen; returns immediately otherwise.
// Caller must ensure config.model_config.use_dsa is true before calling.
void PrepareDSAExpandedInputLength(DSAAttentionMeta& dsa_meta, DSASharedAttentionMeta& dsa_shared,
                                   const AttentionMeta& input, const AttentionBuilderConfig& config);

// Build and upload expanded_block_table for the DSA MTP decode path.
// Only active when input.q_seq_len > kDeepGemmMaxQSeqLen; returns immediately otherwise.
// max_num_blocks_per_query is derived internally from input.dp_reqs.
// Caller must ensure config.model_config.use_dsa is true before calling.
void PrepareDSAExpandedBlockTable(DSAAttentionMeta& dsa_meta, DSASharedAttentionMeta& dsa_shared,
                                  const AttentionMeta& input, const AttentionBuilderConfig& config);

}  // namespace ksana_llm
