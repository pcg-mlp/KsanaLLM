/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/dsa_shared_attention_meta.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/deepseek_deepgemm_wrapper.h"
#  include "csrc/utils/nvidia/cuda_utils.h"
#endif

namespace ksana_llm {

#ifdef ENABLE_CUDA
using ksana_llm::nvidia::kDeepGemmMaxQSeqLen;
#else
constexpr int kDeepGemmMaxQSeqLen = 2;
#endif

void DSASharedAttentionMeta::Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                  int layer_num_on_node, size_t max_table_block_num, size_t max_token_num,
                                  size_t decode_token_num_threshold) {
  if (!model_config.use_dsa) {
    return;
  }

  const size_t max_batch_size = runtime_config.max_batch_size;

  store_out_loc = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  layer_indexer_kv_cache_ptr =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1 + static_cast<size_t>(layer_num_on_node * 2)});

  int num_sms = 0;
#ifdef ENABLE_CUDA
  num_sms = llm_kernels::utils::GetSMCount();
#endif
  for (size_t q_len = 1; q_len <= decode_token_num_threshold; ++q_len) {
    paged_schedule_metas.emplace_back(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                                      std::vector<size_t>{static_cast<size_t>(num_sms + 1), 2, 1});
  }

  if (decode_token_num_threshold > kDeepGemmMaxQSeqLen) {
    expanded_input_length =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size * decode_token_num_threshold});
    expanded_block_table = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                                  {max_batch_size * max_table_block_num * decode_token_num_threshold});
  }
}

}  // namespace ksana_llm
