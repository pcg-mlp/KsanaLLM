/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

struct InputRefitCPUTensor {
  // Tensor to hold pairs(pos, data_length) of positions for input_refit on the CPU.
  Tensor pos_pair_tensor;
  Tensor emb_fp32_ptr_tensor;
};

// Convert input ids to expected format.
class ModelInput {
 public:
  ModelInput(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);
  ~ModelInput();

  // Parse forward request.
  void ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs);

 private:
  // Prepare the kv cache blocks, in CSR format.
  void PrepareKVCacheBlocks(const std::vector<ForwardRequest>& forward_reqs, size_t begin_idx, size_t end_idx,
                            size_t total_block_num);

  void PreparePrefillPositionIds(const std::vector<ForwardRequest>& forward_reqs);

  void PrepareDecodePositionIds(const std::vector<ForwardRequest>& forward_reqs);

  void PreparePrefillInputIds(const std::vector<ForwardRequest>& forward_reqs);

  void PrepareDecodeInputIds(const std::vector<ForwardRequest>& forward_reqs);

  void PrepareInputRefit(const std::vector<ForwardRequest>& forward_reqs);

#ifdef ENABLE_ACL_ATB
  void PrepareATBKVCache(const std::vector<ForwardRequest>& forward_reqs, bool is_context_stage);
#endif

 public:
  // The input batch size.
  size_t batch_size;

  // The context total sequence length.
  size_t context_total_seq_len = 0;

  // ContextDecode reqs num.
  size_t context_num = 0;

  // Decode reqs num.
  size_t decode_num = 0;

  // The total prefix length.
  size_t total_prefix_len = 0;

  // The total block numbe.
  size_t context_total_block_num = 0;
  size_t decode_total_block_num = 0;

  // The max tokens.
  size_t context_max_tokens = 0;
  size_t decode_max_tokens = 0;

  // The cache offset list.
  std::vector<int> kv_cache_offset_list;
  std::vector<size_t> input_offset_list_uint64;
  std::vector<size_t> input_prefix_list_uint64;
  std::vector<int> input_ids_cpu;

  // The infer stage, context decode or decode.
  InferStage infer_stage;

  // The input ids, int32
  Tensor input_ids;

  // The ids offset tensor, uint64
  Tensor input_offset_uint64_tensor;
  // If use_logits_custom_length is true, use logits_custom_length_uint64_tensor instead of input_offset_uint64_tensor
  // for calculation.
  Tensor logits_custom_length_uint64_tensor;
  // Flag to indicate if custom logits length is used.
  bool use_logits_custom_length = false;
  Tensor input_tokens_int32_tensor;

  // Indicate the corresponding index position of the input during rotary_embedding kernel.
  Tensor rotary_embedding_pos;

  // Due to the optimization of PrefixCaching for computation reuse, a mask is used during
  // rotary_embedding computation to avoid multiple executions of rotary_embedding on the prefix block.
  Tensor rotary_embedding_mask;

  // The input's prefix length
  Tensor input_prefix_uint64_tensor;
  // If use_logits_custom_length is true, use logits_length_prefix_uint64_tensor instead of input_prefix_uint64_tensor
  // for calculation.
  Tensor logits_length_prefix_uint64_tensor;

  // Input offset sequence and input prefix sequence on the CPU
  std::vector<int> input_offset_list;
  std::vector<int> input_prefix_list;

  Tensor kv_cache_buffer;
  Tensor kv_cache_offset_tensor;
  Tensor kv_list;
  std::vector<void*> cpu_kv_list;

  // Tensors to hold pairs(pos, data_length) and embeddings ptr of positions for input_refit on the CPU.
  InputRefitCPUTensor cpu_input_refit_tensor;

  Event kvcache_offset_event;
  Event rotary_embedding_event;
  Event input_ids_event;

#ifdef ENABLE_ACL_ATB
  // record all reqs token number on host, shape: [batch_size]
  Tensor seq_len_host;
  // Tensor to save kv cache base. detail doc please refer:
  // docs/Technology/kvcache-relationship-between-ascend-atb-and-ksana.md shape: [total_k/v_blocks, block_token_num,
  // kv_head_num, head_dim]
  Tensor k_cache_blocks_base;
  Tensor v_cache_blocks_base;

  // for prefill stage: layers_slot_mapping shape is [num_layers, all_reqs_tokens_num]
  // for decode stage: layers_block_table shape is [num_layers, batch_size]
  std::vector<int32_t> layers_slot_mapping_host;
  Tensor layers_slot_mapping;

  // only used for decode stage: layers_block_table shape is [num_layers, batch_size * max_num_blocks_per_query]
  std::vector<int32_t> layers_block_table_host;
  Tensor layers_block_table;

  // since layer's forward only support Tensor as input (nothing to do with karlluo), such crappy design ignore runtime
  // attribute, so we need a tensor to be attribute.
  // shape: [2]; 0: layers_slot_mapping_dim_1; 1: max_num_blocks_per_query
  Tensor atb_attention_attr;
#endif

 private:
  ModelConfig model_config_;
  int rank_;
  std::shared_ptr<Context> context_;

  int block_size_;
  size_t max_batch_size_;
  size_t max_token_num_;
  int num_layer_;
};

}  // namespace ksana_llm
