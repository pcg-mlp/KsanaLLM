/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/layers/add_layer.h"
#include "numerous_llm/layers/base_layer.h"
#include "numerous_llm/layers/emb_lookup_layer.h"
#include "numerous_llm/layers/flash_attention_layer.h"
#include "numerous_llm/layers/layernorm_layer.h"
#include "numerous_llm/layers/matmul_layer.h"
#include "numerous_llm/layers/nccl_all_reduce_sum_layer.h"
#include "numerous_llm/layers/paged_attention_layer.h"
#include "numerous_llm/layers/rotary_embedding_layer.h"
#include "numerous_llm/layers/silu_mul_layer.h"
#include "numerous_llm/layers/split_layer.h"
#include "numerous_llm/models/base/base_model.h"
#include "numerous_llm/models/llama/llama_weight.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/tensor.h"
#include "numerous_llm/utils/utils.h"

namespace numerous_llm {

constexpr int RUNTIME_BUFFER_INVALID_IDX = -1;

template <typename T>
class Llama : public BaseModel {
 public:
  Llama(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context);
  ~Llama();

  float* GetLogitsPtr();

  // The prefill stage.
  Status ContextDecode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                       std::vector<ForwardRequest>& forward_reqs);

  // The decode stage.
  Status Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

 protected:
  Status CreateTensor(Tensor& tensor, const std::vector<size_t>& shape, const DataType data_type, int& idx,
                      const bool is_use_buffer = false);
  Status DestroyTensor(Tensor& tensor);
  Status FreeBuffer(Tensor& tensor, const int block_idx);

  // if buffer can be allocated, then return the index,
  // else return -1
  int GetAvaliableBufferIdx();

  // refer to vllm/model_executor/models/llama.py -> class LlamaAttention
  Status ContextDecodeAttention(const int layer_idx, const size_t total_input_token_num,
                                std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                                const Tensor& ids_positions_input, const Tensor& input_ids_seq_len,
                                const Tensor& hidden_states);

  ModelConfig model_config_;

  std::shared_ptr<EmbLookupLayer> emb_lookup_layer_;
  std::shared_ptr<LayernormLayer> layernorm_layer_;
  std::shared_ptr<RotaryEmbeddingLayer> rope_layer_;
  std::vector<std::shared_ptr<FlashAttentionLayer>> flash_attention_layer_;
  std::vector<std::shared_ptr<PagedAttentionLayer>> paged_attention_layer_;
  std::shared_ptr<NcclAllReduceSumLayer> nccl_all_reduce_sum_layer_;
  std::shared_ptr<AddLayer> add_layer_;
  std::shared_ptr<SiluMulLayer> silu_mul_layer_;
  std::shared_ptr<MatMulLayer> matmul_layer_;
  std::shared_ptr<SplitLayer> split_layer_;

  int num_layer_;
  int rank_;
  size_t hidden_units_;
  int pad_token_id_;
  uint32_t vocab_size_;
  float layernorm_eps_;
  DataType weight_data_type_;
  int max_position_embeddings_{2048};

  std::vector<Tensor> empty_tensor_vec_;

  // to mark buffer is used or avaliable
  size_t status_pos_{0ul};
  std::vector<bool> runtime_buffers_status_;
  std::vector<Tensor> runtime_buffers_;
  Tensor kv_cache_buffer_;

  std::shared_ptr<Context> context_{nullptr};

  std::string saved_dir = "/model/llama-ft/7B/nllm/";
};

}  // namespace numerous_llm
