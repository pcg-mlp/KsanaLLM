/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/add_layer.h"
#include "ksana_llm/layers/assemble_last_token_layer.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/cast_layer.h"
#include "ksana_llm/layers/custom_all_reduce_sum_layer.h"
#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/layers/matmul_layer.h"
#include "ksana_llm/layers/nccl_all_reduce_sum_layer.h"
#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/layers/silu_mul_layer.h"

#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/models/llama/llama_weight.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T>
class Llama : public BaseModel {
 public:
  Llama(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context);
  ~Llama();

  float* GetLogitsPtr();

  // The prefill stage.
  Status ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

  // The decode stage.
  Status Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

 private:
  using BaseModel::context_;
  using BaseModel::logits_tensor_;
  using BaseModel::rank_;
  using BaseModel::use_custom_all_reduce_;

  std::shared_ptr<EmbLookupLayer> emb_lookup_layer_;
  std::shared_ptr<LayernormLayer> layernorm_layer_;
  std::vector<std::shared_ptr<FlashAttentionLayer>> flash_attention_layers_;
  std::vector<std::shared_ptr<PagedAttentionLayer>> paged_attention_layers_;
  std::shared_ptr<NcclAllReduceSumLayer> nccl_all_reduce_sum_layer_;
  std::shared_ptr<CustomAllReduceSumLayer> custom_all_reduce_sum_layer_0_;
  std::shared_ptr<AddLayer> add_layer_;
  std::shared_ptr<SiluMulLayer> silu_mul_layer_;
  std::shared_ptr<MatMulLayer> matmul_layer_;
  std::shared_ptr<AssembleLastTokenLayer> assemble_last_token_layer_;
  std::shared_ptr<CastLayer> cast_layer_;

  int num_layer_;
  int max_seq_len_;
  int max_batch_size_;
  size_t hidden_units_;
  int pad_token_id_;
  uint32_t vocab_size_;
  float layernorm_eps_;
  DataType weight_data_type_;
  int block_token_num_;
  int block_size_;
  size_t max_token_num_{0ul};
  bool qkv_add_bias_;

  Tensor reduce_tensor_;
  Tensor rank_tensor_0_;
  Tensor tensor_buffer_0_;
  Tensor tensor_buffer_1_;
  Tensor tensor_buffer_2_;
  Tensor up_matmul_tensor_buffer_;
  Tensor kv_cache_buffer_;
  Tensor kv_cache_offset_tensor_;
  Tensor kv_list_;
  Tensor input_ids_;
  Tensor input_offset_int32_tensor_;
  Tensor input_offset_uint64_tensor_;
  Tensor input_tokens_int32_tensor_;
  Tensor rotary_embedding_pos_;
  Tensor forward_shape_;
  Tensor cos_sin_cache_tensor_;

  Event kvcache_offset_event_;
  Event rotary_embedding_event_;
  Event input_ids_event_;
  Event nccl_finish_event_;
  Event compute_ready_event_;
  Event logits_transfer_event_;

 private:
  void PrepareKVCache(const size_t batch_size, size_t& total_seq_len, size_t& total_block_num,
                      const std::vector<ForwardRequest>& forward_reqs, std::vector<int>& kv_cache_offset_list,
                      Stream& stream, Event& event, bool is_context_stage = false);

  void PrepareContextRotaryEmbeddingPos(const size_t batch_size, const size_t total_seq_len,
                                        const std::vector<ForwardRequest>& forward_reqs, Stream& stream, Event& event);

  void PrepareRotaryEmbeddingPos(const size_t batch_size, const std::vector<ForwardRequest>& forward_reqs,
                                 Stream& stream, Event& event);

  void PrepareContextInputIds(const size_t batch_size, const size_t total_seq_len, int& max_tokens,
                              const std::vector<ForwardRequest>& forward_reqs, Stream& stream, Event& event);

  void PrepareInputIds(const size_t batch_size, int& max_tokens, const std::vector<ForwardRequest>& forward_reqs,
                       Stream& stream, Event& event);

  void CopyToLogistBuffer(const size_t batch_size, std::vector<ForwardRequest>& forward_reqs,
                          std::vector<Tensor>& logits_float);

  // refer to
  // https://github.com/huggingface/transformers/blob/00c1d87a7d5c8dfb4554370983b5a3f7c069edd7/src/transformers/models/llama/modeling_llama.py#L257
  Status LlamaAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight, Tensor& hidden_states,
                        std::vector<Tensor>& output_0, std::vector<Tensor>& output_1, std::vector<Tensor>& output_2,
                        const bool is_context_stage);

  // refer to
  // https://github.com/huggingface/transformers/blob/00c1d87a7d5c8dfb4554370983b5a3f7c069edd7/src/transformers/models/llama/modeling_llama.py#L211
  Status LlamaMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                  Tensor& post_layernorm_output, std::vector<Tensor>& output_0, std::vector<Tensor>& output_1,
                  std::vector<Tensor>& output_2);

  // refer to
  // https://github.com/huggingface/transformers/blob/00c1d87a7d5c8dfb4554370983b5a3f7c069edd7/src/transformers/models/llama/modeling_llama.py#L694
  Status LlamaDecoder(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                      std::vector<Tensor>& temp_buffer_0, std::vector<Tensor>& temp_buffer_1,
                      std::vector<Tensor>& temp_buffer_2, const bool is_context_stage);
};

}  // namespace ksana_llm
