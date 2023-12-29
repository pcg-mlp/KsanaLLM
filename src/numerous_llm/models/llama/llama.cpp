/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/memory_utils.h"

namespace numerous_llm {

constexpr int DEFAULT_RUNTIME_BUFFER_NUM = 3;

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
Status Llama<T>::CreateTensor(Tensor& tensor, size_t total_bytes, DataType data_type) {
  int block_id;
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(total_bytes, block_id);
  tensor = Tensor(MEMORY_GPU, STORAGE_CONTIGUOUS, data_type,
                  std::vector<size_t>{total_bytes / GetTensorTypeSize(data_type)}, std::vector<int>{block_id});
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
Llama<T>::~Llama() {
  GetBlockManager()->SetDeviceId(rank_);

  for (int runtime_buffer_idx = 0; runtime_buffer_idx < DEFAULT_RUNTIME_BUFFER_NUM; ++runtime_buffer_idx) {
    DestroyTensor(runtime_buffers_[runtime_buffer_idx]);
  }
  DestroyTensor(kv_cache_buffer_);
}

template <typename T>
Llama<T>::Llama(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context)
    : context_(context), model_config_(model_config) {
  // 解析 Model Config
  num_layer_ = model_config.num_layer;
  rank_ = rank;
  // TODO: 目前所有层都使用这个 dtype
  weight_data_type_ = model_config.weight_data_type;
  vocab_size_ = model_config.vocab_size;
  float layernorm_eps_ = model_config.layernorm_eps;
  int hidden_units = model_config.size_per_head * model_config.head_num;
  hidden_units_ = static_cast<size_t>(hidden_units);
  // 1: KV 一块空间
  // 2: 运行时中间空间
  // 3: 矩阵计算需要一块空间

  // input_ids: [max_b, max_s]
  // TODO: 数据从 model_config 获取
  const int max_b = model_config.default_batch_size;
  const int max_s = model_config.max_token_num;
  const size_t dtype_size = sizeof(float);

  runtime_buffers_.reserve(DEFAULT_RUNTIME_BUFFER_NUM);
  for (int runtime_buffer_idx = 0; runtime_buffer_idx < DEFAULT_RUNTIME_BUFFER_NUM; ++runtime_buffer_idx) {
    Tensor tmp_tensor;
    CreateTensor(tmp_tensor, max_b * max_s * hidden_units * dtype_size * 2, GetTensorType<float>());
    runtime_buffers_.emplace_back(std::move(tmp_tensor));
  }
  CreateTensor(kv_cache_buffer_, num_layer_ * max_b * max_s * 2 * hidden_units * dtype_size, GetTensorType<float>());

  emb_lookup_layer_ = std::make_shared<EmbLookupLayer>();
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  nccl_all_reduce_sum_layer_ = std::make_shared<NcclAllReduceSumLayer>();
  add_layer_ = std::make_shared<AddLayer>();
  silu_mul_layer_ = std::make_shared<SiluMulLayer>();
  matmul_layer_ = std::make_shared<MatMulLayer>();
  emb_lookup_layer_->Init({}, context_, rank_);
  layernorm_layer_->Init({layernorm_eps_}, context_, rank_);
  nccl_all_reduce_sum_layer_->Init({}, context_, rank_);
  add_layer_->Init({}, context_, rank_);
  silu_mul_layer_->Init({}, context_, rank_);
  matmul_layer_->Init({}, context_, rank_);
  flash_attention_layer_.resize(num_layer_);
  paged_attention_layer_.resize(num_layer_);
  for (int idx = 0; idx < num_layer_; ++idx) {
    flash_attention_layer_[idx] = std::make_shared<FlashAttentionLayer>();
    paged_attention_layer_[idx] = std::make_shared<PagedAttentionLayer>();
    flash_attention_layer_[idx]->Init({idx, 2048}, context_, rank_);
    paged_attention_layer_[idx]->Init({idx, 2048}, context_, rank_);
  }
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
  // inputs tokens consist with two tensors as same as CSR
  // 1. input_ids tensor:
  //    [1, 2, 3, 4, 5, 6, 7]
  // 2. input offset tensor
  //    [0, 3, 7]
  // each worker threads get inputs tokens
  // handle it and copy it to GPU async
  // input_ids tensor
  void* input_ids_ptr = runtime_buffers_[0].GetPtr<void>();
  size_t input_offset = 0ul;
  std::vector<size_t> input_offset_list(batch_size + 1, 0ul);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t input_token_length = req_input->size();
    total_input_token_num += input_token_length;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<int*>(input_ids_ptr) + input_offset, req_input->data(),
                               input_token_length * sizeof(int), cudaMemcpyHostToDevice,
                               context_->GetH2DStreams()[rank_]));
    input_offset += input_token_length;
    input_offset_list[idx + 1] = input_offset;
  }

  // input offset tensor
  void* input_offset_ptr = runtime_buffers_[1].GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(input_offset_ptr, input_offset_list.data(), input_offset_list.size() * sizeof(size_t),
                             cudaMemcpyHostToDevice, context_->GetH2DStreams()[rank_]));

  // overlap tensor create and data copy time
  Tensor input_ids;
  input_ids.shape = {total_input_token_num};
  input_ids.dtype = TYPE_INT32;
  input_ids.blocks = runtime_buffers_[0].blocks;
  Tensor input_offset_tensor;
  input_offset_tensor.shape = {batch_size + 1};
  input_offset_tensor.dtype = TYPE_UINT64;
  input_offset_tensor.blocks = runtime_buffers_[1].blocks;

  CUDA_CHECK(cudaStreamSynchronize(context_->GetH2DStreams()[rank_]));

  // Embedding forward
  Tensor embedding_weight = base_weight->GetModelWeights("gather_embedding");
  Tensor embedding_output;
  embedding_output.shape = {total_input_token_num, hidden_units_};
  embedding_output.dtype = GetTensorType<T>();
  embedding_output.blocks = runtime_buffers_[2].blocks;
  std::vector<Tensor> embedding_outputs = {embedding_output};
  STATUS_CHECK_RETURN(
      emb_lookup_layer_->Forward({input_ids, input_offset_tensor, embedding_weight}, embedding_outputs));
  // runtime_buffers 0, 1 can be reused, 2 in used

  // LlamaDecoder forward
  for (int layer_num = 0; layer_num < num_layer_; ++layer_num) {
    // input layernorm
    Tensor input_layernorm_weight = base_weight->GetModelWeights(fmt::format("{}.input_layernorm", layer_num));
    // resue input ids buffer
    Tensor layernorm_output;
    layernorm_output.shape = {total_input_token_num, hidden_units_};
    layernorm_output.dtype = GetTensorType<T>();
    layernorm_output.blocks = runtime_buffers_[0].blocks;
    std::vector<Tensor> layernorm_outputs = {layernorm_output};
    STATUS_CHECK_RETURN(layernorm_layer_->Forward({embedding_output, input_layernorm_weight}, layernorm_outputs));
    // runtime_buffers 1, 2 can be reused, 0 in used

    // attn project matmul
    Tensor attn_proj_weight = base_weight->GetModelWeights(fmt::format("{}.attention.query_key_value", layer_num));
    Tensor attn_proj_output;
    attn_proj_output.shape = {total_input_token_num, attn_proj_weight.shape[1]};
    attn_proj_output.dtype = GetTensorType<T>();
    attn_proj_output.blocks = runtime_buffers_[1].blocks;
    std::vector<Tensor> attn_proj_outputs = {attn_proj_output};
    STATUS_CHECK_RETURN(matmul_layer_->Forward({layernorm_output, attn_proj_weight}, attn_proj_outputs));
    // runtime_buffers 0, 2 can be reused, 1 in used
  }

  CUDA_CHECK(cudaStreamSynchronize(context_->GetComputeStreams()[rank_]));

  return Status();
}

template <typename T>
Status Llama<T>::Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest>& forward_reqs) {
  NLLM_LOG_INFO << "llama decode stage inference";
  return Status();
}

template class Llama<float>;
template class Llama<half>;

}  // namespace numerous_llm
