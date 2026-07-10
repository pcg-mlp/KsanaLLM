/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>

#include "device/device_utils.h"

namespace ksana_llm {

// Hardware-neutral dispatch wrapper for the fused Dsv4 MQA meta kernel.
// Computes c128_block_indices, c128_topk_lengths, AND swa_block_indices in a single launch.
// Implementation in prepare_dsv4_mqa_meta_dispatch.cpp selects the backend at compile time.
void PrepareDsv4MqaMeta(const int32_t* cu_extend_lens, const int32_t* cu_prefix_lens, const int32_t* block_table,
                        const int32_t* raw_out_loc, const int32_t* swa_prefix_tail_block_indices, int batch_size,
                        int block_token_num, int block_table_stride, int max_extend_len, int32_t* swa_block_indices,
                        int32_t* c128_block_indices, int32_t* c128_topk_lengths, int swa_size,
                        int c128_block_indices_stride, bool fill_topk_with_stride, bool noncausal, Stream& stream);

}  // namespace ksana_llm
