/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/add_layer.h"
#include "ksana_llm/layers/assemble_last_token_layer.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/cast_layer.h"
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

 protected:
  Status CreateTensor(Tensor& tensor, size_t length);
  Status DestroyTensor(Tensor& tensor);

  std::shared_ptr<EmbLookupLayer> emb_lookup_layer_;
  std::shared_ptr<LayernormLayer> layernorm_layer_;
  std::vector<std::shared_ptr<FlashAttentionLayer>> flash_attention_layer_;
  std::vector<std::shared_ptr<PagedAttentionLayer>> paged_attention_layer_;
  std::shared_ptr<NcclAllReduceSumLayer> nccl_all_reduce_sum_layer_;
  std::shared_ptr<AddLayer> add_layer_;
  std::shared_ptr<SiluMulLayer> silu_mul_layer_;
  std::shared_ptr<MatMulLayer> matmul_layer_;
  std::shared_ptr<AssembleLastTokenLayer> assemble_last_token_layer_;
  std::shared_ptr<CastLayer> cast_layer_;
  int num_layer_;
  int rank_;
  int max_seq_len_;
  int max_batch_size_;
  size_t hidden_units_;
  int pad_token_id_;
  uint32_t vocab_size_;
  float layernorm_eps_;
  DataType weight_data_type_;
  int block_token_num_;
  int block_size_;
  Tensor tmp_tensor_0, tmp_tensor_1, tmp_tensor_2;
  Tensor up_matmul_tensor;
  Tensor kv_cache_buffer_;
  Tensor logits_tensor_;
  Tensor kv_cache_offset_tensor;
  Tensor kv_list;
  Tensor input_ids, input_offset_int32_tensor, input_offset_uint64_tensor;
  Tensor input_tokens_int32_tensor;
  Tensor rotary_embedding_pos;
  Tensor forward_shape;
  Tensor cos_sin_cache_tensor;
  std::shared_ptr<Context> context_{nullptr};

  std::string saved_dir = "/model/llama-ft/7B/nllm/";

 private:
  cudaEvent_t kvcache_offset_event_;
  cudaEvent_t rotary_embedding_event_;
  cudaEvent_t input_ids_event_;
  cudaEvent_t nccl_finish_event_;
  cudaEvent_t compute_ready_event_;
  cudaEvent_t logits_transfer_event_;

 private:
  void PrepareKVCache(const size_t batch_size, size_t& total_seq_len, size_t& total_block_num,
                      const std::vector<ForwardRequest>& forward_reqs, std::vector<int>& kv_cache_offset_list,
                      cudaStream_t& stream, cudaEvent_t& event);

  void PrepareContextRotaryEmbeddingPos(const size_t batch_size, const size_t total_seq_len,
                                        const std::vector<ForwardRequest>& forward_reqs, cudaStream_t& stream,
                                        cudaEvent_t& event);

  void PrepareRotaryEmbeddingPos(const size_t batch_size, const std::vector<ForwardRequest>& forward_reqs,
                                 cudaStream_t& stream, cudaEvent_t& event);

  void PrepareContextInputIds(const size_t batch_size, const size_t total_seq_len, int& max_tokens,
                              const std::vector<ForwardRequest>& forward_reqs, cudaStream_t& stream,
                              cudaEvent_t& event);

  void PrepareInputIds(const size_t batch_size, int& max_tokens, const std::vector<ForwardRequest>& forward_reqs,
                       cudaStream_t& stream, cudaEvent_t& event);

  void CopyToLogistBuffer(const size_t batch_size, cudaEvent_t& compute_ready_event_, cudaStream_t& compute_stream,
                          cudaStream_t& d2d_stream, std::vector<ForwardRequest>& forward_reqs,
                          std::vector<Tensor>& logits_float);
};

}  // namespace ksana_llm
