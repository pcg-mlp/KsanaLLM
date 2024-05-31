/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/tensor.h"

#include <vector>

namespace ksana_llm {

// Convert input ids to expected format.
class ModelInput {
 public:
  ModelInput(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);
  ~ModelInput();

  // Parse forward request.
  void ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs, bool is_context_stage);

 private:
  // Prepare the kv cache blocks, in CSR format.
  void PrepareKVCacheBlocks(const std::vector<ForwardRequest>& forward_reqs);

  void PreparePrefillPositionIds(const std::vector<ForwardRequest>& forward_reqs);

  void PrepareDecodePositionIds(const std::vector<ForwardRequest>& forward_reqs);

  void PreparePrefillInputIds(const std::vector<ForwardRequest>& forward_reqs);

  void PrepareDecodeInputIds(const std::vector<ForwardRequest>& forward_reqs);

  void PrepareSubinput(const std::vector<ForwardRequest>& forward_reqs);

 public:
  // The input batch size.
  size_t batch_size;

  // The total sequence length.
  size_t total_seq_len = 0;

  // The total block numbe.
  size_t total_block_num = 0;

  // The max tokens.
  size_t max_tokens = 0;

  // The cache offset list.
  std::vector<int> kv_cache_offset_list;

  // The infer stage, context decode or decode.
  InferStage infer_stage;

  // The input ids, int32
  Tensor input_ids;

  // The ids offset tensor, uint64
  Tensor input_offset_uint64_tensor;
  // If prompt props is returned, use prompt_probs_offset_uint64_tensor instead of input_offset_uint64_tensor for
  // calculation.
  Tensor prompt_probs_offset_uint64_tensor;
  bool use_prompt_probs_offset = false;
  Tensor input_tokens_int32_tensor;
  Tensor rotary_embedding_pos;

  Tensor kv_cache_buffer;
  Tensor kv_cache_offset_tensor;
  Tensor kv_list;

  // Tensor to hold pairs(pos, data_length) of positions for subinputs on the CPU.
  Tensor cpu_subinput_pos_pair_tensor;
  Tensor cpu_subinput_emb_fp32_ptr_tensor;

  Event kvcache_offset_event;
  Event rotary_embedding_event;
  Event input_ids_event;

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
