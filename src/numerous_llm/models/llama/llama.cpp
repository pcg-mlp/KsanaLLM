/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/memory_utils.h"

namespace numerous_llm {

constexpr int DEFAULT_RUNTIME_BUFFER_NUM = 9;
constexpr int DEFAULT_MAX_POS_EMBEDDING_SIZE = 2048;

size_t GetTensorTypeSize(DataType data_type) {
  if (TYPE_FP32 == data_type) {
    return sizeof(float);
  } else if (TYPE_FP16 == data_type) {
    return sizeof(half);
  }
#ifdef ENABLE_BF16
  else if (TYPE_BF16 == data_type) {
    return sizeof(__nv_bfloat16);
  }
#endif
#ifdef ENABLE_FP8
  else if (TYPE_FP8_E4M3 == data_type) {
    return sizeof(__nv_fp8_e4m3);
  }
#endif
  else if (TYPE_INT32 == data_type) {
    return sizeof(int32_t);
  } else if (TYPE_INT8 == data_type) {
    return sizeof(int8_t);
  } else if (TYPE_UINT32 == data_type) {
    return sizeof(uint32_t);
  } else if (TYPE_UINT64 == data_type) {
    return sizeof(uint64_t);
  } else if (TYPE_BOOL == data_type) {
    return sizeof(bool);
  } else if (TYPE_BYTES == data_type) {
    return sizeof(char);
  } else if (TYPE_POINTER == data_type) {
    return sizeof(std::intptr_t);
  } else {
    // for TYPE_INVALID
    return 0ul;
  }
  return 0ul;
}

template <typename T>
Llama<T>::Llama(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context)
    : context_(context), model_config_(model_config) {
  rank_ = rank;
  num_layer_ = model_config_.num_layer;
  weight_data_type_ = model_config_.weight_data_type;
  vocab_size_ = model_config_.vocab_size;
  float layernorm_eps_ = model_config_.layernorm_eps;
  int hidden_units = model_config_.size_per_head * model_config_.head_num;
  hidden_units_ = static_cast<size_t>(hidden_units);
  // 1: KV 一块空间
  // 2: 运行时中间空间
  // 3: 矩阵计算需要一块空间

  // input_ids: [max_b, max_s]
  const int max_b = model_config.default_batch_size;
  const int max_s = model_config.max_token_num;

  runtime_buffers_.reserve(DEFAULT_RUNTIME_BUFFER_NUM);
  runtime_buffers_status_.resize(DEFAULT_RUNTIME_BUFFER_NUM, true);
  for (int runtime_buffer_idx = 0; runtime_buffer_idx < DEFAULT_RUNTIME_BUFFER_NUM; ++runtime_buffer_idx) {
    Tensor tmp_tensor;
    int block_id = -1;
    CreateTensor(tmp_tensor,
                 {static_cast<size_t>(max_b), static_cast<size_t>(max_s), 3ul, static_cast<size_t>(hidden_units)},
                 GetTensorType<float>(), block_id, false);
    runtime_buffers_.emplace_back(std::move(tmp_tensor));
  }
  int block_id = -1;
  CreateTensor(kv_cache_buffer_,
               {static_cast<size_t>(num_layer_), static_cast<size_t>(max_b), static_cast<size_t>(max_s),
                static_cast<size_t>(hidden_units)},
               GetTensorType<float>(), block_id, false);

  // create instance
  emb_lookup_layer_ = std::make_shared<EmbLookupLayer>();
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  nccl_all_reduce_sum_layer_ = std::make_shared<NcclAllReduceSumLayer>();
  add_layer_ = std::make_shared<AddLayer>();
  silu_mul_layer_ = std::make_shared<SiluMulLayer>();
  matmul_layer_ = std::make_shared<MatMulLayer>();
  rope_layer_ = std::make_shared<RotaryEmbeddingLayer>();
  split_layer_ = std::make_shared<SplitLayer>();

  // init
  emb_lookup_layer_->Init({}, context_, rank_);
  layernorm_layer_->Init({layernorm_eps_}, context_, rank_);
  nccl_all_reduce_sum_layer_->Init({}, context_, rank_);
  add_layer_->Init({}, context_, rank_);
  silu_mul_layer_->Init({}, context_, rank_);
  matmul_layer_->Init({}, context_, rank_);
  rope_layer_->Init(
      {max_position_embeddings_, model_config_.rotary_embedding, model_config_.rope_theta, model_config_.size_per_head,
       model_config_.head_num, model_config_.num_key_value_heads, /*is_neox*/ true},
      context_, rank_);
  flash_attention_layer_.resize(num_layer_);
  paged_attention_layer_.resize(num_layer_);
  NLLM_LOG_INFO << static_cast<int>(model_config_.head_num) << ", " << static_cast<int>(model_config_.size_per_head);
  for (int idx = 0; idx < num_layer_; ++idx) {
    flash_attention_layer_[idx] = std::make_shared<FlashAttentionLayer>();
    paged_attention_layer_[idx] = std::make_shared<PagedAttentionLayer>();
    flash_attention_layer_[idx]->Init({idx, DEFAULT_MAX_POS_EMBEDDING_SIZE, static_cast<int>(model_config_.head_num),
                                       static_cast<int>(model_config_.size_per_head)},
                                      context_, rank_);
    paged_attention_layer_[idx]->Init({idx, DEFAULT_MAX_POS_EMBEDDING_SIZE, static_cast<int>(model_config_.head_num),
                                       static_cast<int>(model_config_.size_per_head)},
                                      context_, rank_);
  }
}

template <typename T>
Llama<T>::~Llama() {
  GetBlockManager()->SetDeviceId(rank_);

  for (int runtime_buffer_idx = 0; runtime_buffer_idx < DEFAULT_RUNTIME_BUFFER_NUM; ++runtime_buffer_idx) {
    DestroyTensor(runtime_buffers_[runtime_buffer_idx]);
  }
  DestroyTensor(kv_cache_buffer_);
}

template <typename T>
float* Llama<T>::GetLogitsPtr() {
  return nullptr;
}

template <typename T>
Status Llama<T>::ContextDecode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                               std::vector<ForwardRequest>& forward_reqs) {
  NLLM_LOG_INFO << "llama context decode stage inference";
  size_t batch_size = forward_reqs.size();
  size_t total_input_token_num = 0ul;

  // prepare inputs
  // for example:
  //   2 prompt: "Today is good."
  //             "Dog is cute."
  // inputs tokens: [[1, 2, 3],
  //                 [4, 5, 6, 7]]
  // inputs tokens consist with three tensors as same as CSR
  // 1. input_ids tensor use buffer id 0
  //    [1, 2, 3, 4, 5, 6, 7]
  // 2. input offset tensor use buffer id 1
  //    [0, 3, 7]
  // 3. ids position tensor (used by rope) use buffer id 2
  //    [1, 2, 3, 1, 2, 3, 4]
  // 4. input length tensor (used by flash attention) use buffer id 3
  //    [3, 4]
  // each worker threads get inputs tokens
  // handle it and copy it to GPU async
  // input_ids tensor
  Tensor input_ids;
  int input_ids_buffer_idx;
  STATUS_CHECK_RETURN(CreateTensor(input_ids, {total_input_token_num}, GetTensorType<int32_t>(), input_ids_buffer_idx,
                                   /*is_use_buffer*/ true));
  void* input_ids_ptr = input_ids.GetPtr<void>();
  size_t input_offset = 0ul;
  std::vector<size_t> input_offset_list(batch_size + 1, 0ul);
  std::vector<int32_t> input_len(batch_size + 1, static_cast<int32_t>(0));
  for (size_t idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t input_token_length = req_input->size();
    total_input_token_num += input_token_length;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<int*>(input_ids_ptr) + input_offset, req_input->data(),
                               input_token_length * sizeof(int), cudaMemcpyHostToDevice,
                               context_->GetH2DStreams()[rank_]));
    input_offset += input_token_length;
    input_offset_list[idx + 1] = input_offset;
    input_len[idx + 1] = static_cast<int32_t>(input_offset);
  }

  // position tensor
  std::vector<int64_t> ids_positions(total_input_token_num, 0ul);
  input_offset = 0ul;
  for (size_t idx = 0; idx < batch_size; ++idx) {
    size_t input_token_length = forward_reqs[idx].output_tokens->size();
    std::iota(ids_positions.begin() + input_offset, ids_positions.begin() + input_offset + input_token_length,
              static_cast<int64_t>(1));
    input_offset += input_token_length;
    input_len[idx] = input_token_length;
  }
  Tensor ids_positions_input;
  int ids_positions_buffer_idx;
  STATUS_CHECK_RETURN(CreateTensor(ids_positions_input, {total_input_token_num}, GetTensorType<int64_t>(),
                                   ids_positions_buffer_idx,
                                   /*is_use_*/ true));
  void* ids_positions_ptr = ids_positions_input.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(ids_positions_ptr, ids_positions.data(), ids_positions.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice, context_->GetH2DStreams()[rank_]));

  // input length tensor
  Tensor input_ids_seq_len;
  int input_ids_seq_len_buffer_idx;
  STATUS_CHECK_RETURN(CreateTensor(input_ids_seq_len, {batch_size}, GetTensorType<int32_t>(),
                                   input_ids_seq_len_buffer_idx, /*is_use_buffer*/ true));
  void* input_ids_seq_len_ptr = input_ids_seq_len.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(input_ids_seq_len_ptr, input_len.data(), input_len.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, context_->GetH2DStreams()[rank_]));

  // input offset tensor
  Tensor input_offset_tensor;
  int input_offset_tensor_idx;
  STATUS_CHECK_RETURN(CreateTensor(input_offset_tensor, {batch_size + 1}, GetTensorType<uint64_t>(),
                                   input_offset_tensor_idx, /*is_use_buffer*/ true));
  void* input_offset_ptr = input_offset_tensor.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(input_offset_ptr, input_offset_list.data(), input_offset_list.size() * sizeof(size_t),
                             cudaMemcpyHostToDevice, context_->GetH2DStreams()[rank_]));

  CUDA_CHECK(cudaStreamSynchronize(context_->GetH2DStreams()[rank_]));

  // Embedding forward
  Tensor embedding_weight = base_weight->GetModelWeights("gather_embedding");
  Tensor embedding_output;
  int embedding_output_buffer_idx;
  STATUS_CHECK_RETURN(CreateTensor(embedding_output, {total_input_token_num, hidden_units_}, GetTensorType<uint64_t>(),
                                   embedding_output_buffer_idx, /*is_use_buffer*/ true));
  std::vector<Tensor> embedding_outputs = {embedding_output};
  STATUS_CHECK_RETURN(
      emb_lookup_layer_->Forward({input_ids, input_offset_tensor, embedding_weight}, embedding_outputs));
  FreeBuffer(input_ids, input_ids_buffer_idx);

  // LlamaDecoder forward
  for (int layer_idx = 0; layer_idx < num_layer_; ++layer_idx) {
    // input layernorm
    Tensor input_layernorm_weight = base_weight->GetModelWeights(fmt::format("{}.input_layernorm", layer_idx));
    // resue input ids buffer
    Tensor layernorm_output;
    int layernorm_output_buffer_idx;
    STATUS_CHECK_RETURN(CreateTensor(layernorm_output, {total_input_token_num, hidden_units_},
                                     GetTensorType<uint64_t>(), layernorm_output_buffer_idx, /*is_use_buffer*/ true));
    std::vector<Tensor> layernorm_outputs = {layernorm_output};
    STATUS_CHECK_RETURN(layernorm_layer_->Forward({embedding_output, input_layernorm_weight}, layernorm_outputs));

    // This code refer to https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
    STATUS_CHECK_RETURN(ContextDecodeAttention(layer_idx, total_input_token_num, base_weight, ids_positions_input,
                                               input_ids_seq_len, layernorm_output));

    FreeBuffer(layernorm_output, layernorm_output_buffer_idx);
  }

  CUDA_CHECK(cudaStreamSynchronize(context_->GetComputeStreams()[rank_]));

  runtime_buffers_status_.resize(runtime_buffers_.size(), true);
  return Status();
}

template <typename T>
Status Llama<T>::Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest>& forward_reqs) {
  NLLM_LOG_INFO << "llama decode stage inference";
  return Status();
}

template <typename T>
Status Llama<T>::CreateTensor(Tensor& tensor, const std::vector<size_t>& shape, const DataType data_type, int& idx,
                              const bool is_use_buffer) {
  if (is_use_buffer) {
    int buffer_idx = GetAvaliableBufferIdx();
    if (buffer_idx == RUNTIME_BUFFER_INVALID_IDX) {
      NLLM_LOG_FATAL << "Can not allocate inference buffer";
      return Status(RET_INVALID_ARGUMENT);
    }
    idx = buffer_idx;
    tensor.dtype = data_type;
    tensor.shape = std::move(shape);
    tensor.storage = STORAGE_CONTIGUOUS;
    tensor.device = MEMORY_GPU;
    // take buffer for inference
    tensor.blocks = runtime_buffers_[buffer_idx].blocks;
  } else {
    GetBlockManager()->SetDeviceId(rank_);
    size_t total_bytes =
        std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) *
        GetTensorTypeSize(data_type);
    GetBlockManager()->AllocateContiguous(total_bytes, idx);
    tensor = Tensor(MEMORY_GPU, STORAGE_CONTIGUOUS, data_type, shape, std::vector<int>{idx});
  }
  return Status();
}

template <typename T>
Status Llama<T>::DestroyTensor(Tensor& tensor) {
  GetBlockManager()->SetDeviceId(rank_);
  const std::vector<int>& block_ids = tensor.GetBlockIds();
  NLLM_CHECK_WITH_INFO(block_ids.size() == 1, "Contiguous must have only one block.");
  return GetBlockManager()->FreeContiguous(block_ids.front());
}

template <typename T>
Status Llama<T>::FreeBuffer(Tensor& tensor, const int block_idx) {
  runtime_buffers_status_[block_idx] = true;
  return Status();
}

template <typename T>
int Llama<T>::GetAvaliableBufferIdx() {
  // runtime_buffers_status_ is small enough,
  // so it is efficient.
  size_t pre_pos_ = status_pos_;
  status_pos_ = (status_pos_ + 1) % runtime_buffers_status_.size();
  if (status_pos_ == pre_pos_) {
    if (runtime_buffers_status_[status_pos_]) {
      runtime_buffers_status_[status_pos_] = false;
      return status_pos_;
    } else {
      return RUNTIME_BUFFER_INVALID_IDX;
    }
  }
  while (status_pos_ != pre_pos_) {
    if (runtime_buffers_status_[status_pos_]) {
      runtime_buffers_status_[status_pos_] = false;
      return status_pos_;
    } else {
      status_pos_ = (status_pos_ + 1) % runtime_buffers_status_.size();
      continue;
    }
  }
  return RUNTIME_BUFFER_INVALID_IDX;
}

template <typename T>
Status Llama<T>::ContextDecodeAttention(const int layer_idx, const size_t total_input_token_num,
                                        std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                                        const Tensor& ids_positions_input, const Tensor& input_ids_seq_len,
                                        const Tensor& hidden_states) {
  // attn project matmul
  // refer to: qkv, _ = self.qkv_proj(hidden_states)
  Tensor attn_proj_weight = base_weight->GetModelWeights(fmt::format("{}.attention.query_key_value", layer_idx));
  Tensor attn_proj_output;
  int attn_proj_output_buffer_idx;
  STATUS_CHECK_RETURN(CreateTensor(attn_proj_output, {total_input_token_num, attn_proj_weight.shape[1]},
                                   GetTensorType<T>(), attn_proj_output_buffer_idx, /*is_use_buffer*/ true));
  std::vector<Tensor> attn_proj_outputs = {attn_proj_output};
  STATUS_CHECK_RETURN(matmul_layer_->Forward({hidden_states, attn_proj_weight}, attn_proj_outputs));

  // TODO(karlluo): slice attn_proj_outputs as key and query, value 3 tensor
  // refer to: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
  // for example: [token_nums, 3 * hidden_units]
  // to:
  //   query: [token_num, hidden_units]
  //   key: [token_num, hidden_units]
  //   value: [token_num, hidden_units]
  Tensor q;
  int q_buffer_idx;
  STATUS_CHECK_RETURN(CreateTensor(q, {total_input_token_num, hidden_units_}, GetTensorType<T>(), q_buffer_idx,
                                   /*is_use_buffer*/ true));
  Tensor k;
  int k_buffer_idx;
  STATUS_CHECK_RETURN(CreateTensor(k, {total_input_token_num, hidden_units_}, GetTensorType<T>(), k_buffer_idx,
                                   /*is_use_buffer*/ true));
  Tensor v;
  int v_buffer_idx;
  STATUS_CHECK_RETURN(CreateTensor(v, {total_input_token_num, hidden_units_}, GetTensorType<T>(), v_buffer_idx,
                                   /*is_use_buffer*/ true));
  std::vector<Tensor> split_outputs = {q, k, v};
  STATUS_CHECK_RETURN(split_layer_->Forward(attn_proj_outputs, split_outputs));
  FreeBuffer(attn_proj_output, attn_proj_output_buffer_idx);

  // refer to: q, k = self.rotary_emb(positions, q, k)
  STATUS_CHECK_RETURN(rope_layer_->Forward({ids_positions_input, q, k}, empty_tensor_vec_));

  Tensor attn_output;
  int attn_output_buffer_idx;
  STATUS_CHECK_RETURN(CreateTensor(attn_output, {total_input_token_num, hidden_units_}, GetTensorType<T>(),
                                   attn_output_buffer_idx, /*is_use_buffer*/ true));
  std::vector<Tensor> attn_outputs = {attn_output};
  // refer to: attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
  STATUS_CHECK_RETURN(flash_attention_layer_[layer_idx]->Forward({q, k, v, input_ids_seq_len}, attn_outputs));
  FreeBuffer(q, q_buffer_idx);
  FreeBuffer(k, k_buffer_idx);
  FreeBuffer(v, v_buffer_idx);

  // attn NcclAllReduceSum with multiple GPU
  Tensor attn_all_reduce_sum_output;
  int attn_all_reduce_sum_output_idx;
  attn_all_reduce_sum_output.shape = {total_input_token_num, hidden_units_};
  STATUS_CHECK_RETURN(CreateTensor(attn_all_reduce_sum_output, {total_input_token_num, hidden_units_},
                                   GetTensorType<T>(), attn_all_reduce_sum_output_idx, /*is_use_buffer*/ true));
  std::vector<Tensor> attn_all_reduce_sum_outputs = {attn_all_reduce_sum_output};
  STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(attn_outputs, attn_all_reduce_sum_outputs));
  FreeBuffer(attn_output, attn_output_buffer_idx);
  FreeBuffer(attn_all_reduce_sum_output, attn_all_reduce_sum_output_idx);

  return Status();
}

template class Llama<float>;
template class Llama<half>;

}  // namespace numerous_llm
