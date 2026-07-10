/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/attention_meta/prepare_dsv4_mqa_meta_dispatch.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/attention_meta/prepare_dsv4_mqa_meta.h"
#else
#  include "utils/logger.h"
#endif

namespace ksana_llm {

void PrepareDsv4MqaMeta(const int32_t* cu_extend_lens, const int32_t* cu_prefix_lens, const int32_t* block_table,
                        const int32_t* raw_out_loc, const int32_t* swa_prefix_tail_block_indices, int batch_size,
                        int block_token_num, int block_table_stride, int max_extend_len, int32_t* swa_block_indices,
                        int32_t* c128_block_indices, int32_t* c128_topk_lengths, int swa_size,
                        int c128_block_indices_stride, bool fill_topk_with_stride, bool noncausal, Stream& stream) {
#ifdef ENABLE_CUDA
  llm_kernels::nvidia::InvokePrepareDsv4MqaMeta(
      cu_extend_lens, cu_prefix_lens, block_table, raw_out_loc, swa_prefix_tail_block_indices, batch_size,
      block_token_num, block_table_stride, max_extend_len, swa_block_indices, c128_block_indices, c128_topk_lengths,
      swa_size, c128_block_indices_stride, fill_topk_with_stride, noncausal, stream.Get());
#else
  KLLM_THROW("PrepareDsv4MqaMeta is only implemented in CUDA");
#endif
}

}  // namespace ksana_llm
