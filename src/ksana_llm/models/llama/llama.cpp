/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/llama/llama.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"

namespace ksana_llm {

template <typename T>
Status Llama<T>::CreateTensor(Tensor& tensor, size_t total_bytes) {
  int block_id;
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(total_bytes, block_id);
  // 此处的 shape 是默认生成的, 2  = sizeof(fp16)
  tensor = Tensor(MEMORY_GPU, STORAGE_CONTIGUOUS, weight_data_type_, std::vector<size_t>{total_bytes / 2},
                  std::vector<int>{block_id});
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

  DestroyTensor(tmp_tensor_0);
  DestroyTensor(tmp_tensor_1);
  DestroyTensor(tmp_tensor_2);
  DestroyTensor(up_matmul_tensor);
  DestroyTensor(kv_cache_buffer_);
  DestroyTensor(logits_tensor_);

  DestroyTensor(input_ids);
  DestroyTensor(input_offset_int32_tensor);
  DestroyTensor(input_offset_uint64_tensor);
  DestroyTensor(kv_list);
  DestroyTensor(forward_shape);
  DestroyTensor(rotary_embedding_pos);
  DestroyTensor(kv_cache_offset_tensor);
  DestroyTensor(cos_sin_cache_tensor);

  // free all cude event
  CUDA_CHECK(cudaEventDestroy(kvcache_offset_event_));
  CUDA_CHECK(cudaEventDestroy(rotary_embedding_event_));
  CUDA_CHECK(cudaEventDestroy(input_ids_event_));
  CUDA_CHECK(cudaEventDestroy(nccl_finish_event_));
  CUDA_CHECK(cudaEventDestroy(compute_ready_event_));
  CUDA_CHECK(cudaEventDestroy(logits_transfer_event_));
}

template <typename T>
Llama<T>::Llama(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context) : context_(context) {
  num_layer_ = model_config.num_layer;
  rank_ = rank;
  GetBlockManager()->SetDeviceId(rank_);
  // TODO: 目前所有层都使用这个 dtype
  weight_data_type_ = model_config.weight_data_type;
  vocab_size_ = model_config.vocab_size;
  float layernorm_eps_ = model_config.layernorm_eps;
  int head_num = model_config.head_num;
  int size_per_head = model_config.size_per_head;
  int hidden_units = size_per_head * head_num;
  int rotary_embedding = model_config.rotary_embedding;
  head_num /= model_config.tensor_para_size;
  int num_key_value_heads = model_config.num_key_value_heads / model_config.tensor_para_size;
  int stride_size = head_num * size_per_head * 3;
  int inter_size = model_config.inter_size;
  int max_position_embeddings = model_config.max_position_embeddings;
  float rope_theta = model_config.rope_theta;
  block_token_num_ = GetBlockManager()->GetBlockTokenNum();
  block_size_ = GetBlockManager()->GetBlockSize();

  // 1: KV 一块空间
  // 2: 运行时中间空间
  // 3: 矩阵计算需要一块空间

  // input_ids: [max_batch_size_, max_s]
  max_batch_size_ = model_config.max_batch_size;
  max_seq_len_ = model_config.max_token_num;
  size_t dtype_size = Tensor::GetTypeSize(weight_data_type_);
  size_t max_token_num = model_config.max_scheduler_token_num;
  NLLM_LOG_DEBUG << fmt::format("Max_Batch_Size = {}, Max Seq Len = {}, Max Token Num = {}", max_batch_size_,
                                max_seq_len_, max_token_num);
  size_t tmp_tensor_size =
      std::max(max_batch_size_ * vocab_size_ * sizeof(float), max_token_num * hidden_units * 3 * dtype_size);
  CreateTensor(tmp_tensor_0, max_token_num * hidden_units * dtype_size * 3);
  CreateTensor(tmp_tensor_1, tmp_tensor_size);
  CreateTensor(tmp_tensor_2, max_token_num * hidden_units * dtype_size * 3);

  size_t up_matmul_tensor_size = max_token_num * dtype_size * std::max(inter_size, hidden_units * 2);
  CreateTensor(up_matmul_tensor, up_matmul_tensor_size);
  CreateTensor(kv_cache_buffer_,
               (size_t)max_seq_len_ * (max_seq_len_ + 511) / 512 * head_num * (size_per_head + 2) * sizeof(float));
  kv_cache_buffer_.shape = {max_seq_len_, (max_seq_len_ + 511) / 512, head_num, size_per_head + 2};
  kv_cache_buffer_.dtype = TYPE_FP32;
  CreateTensor(logits_tensor_, max_batch_size_ * vocab_size_ * sizeof(float));

  CreateTensor(kv_cache_offset_tensor, (max_batch_size_ + 1) * sizeof(int));
  kv_cache_offset_tensor.dtype = TYPE_INT32;

  CreateTensor(cos_sin_cache_tensor, rotary_embedding * max_position_embeddings * sizeof(half));
  half* cos_sin_cache_ptr = cos_sin_cache_tensor.GetPtr<half>();

  GetBlockManager()->SetDeviceId(rank_);
  // TODO: 该步骤先于 Reset device_blocks_Num 发生
  size_t max_block_num = 2048;
  CreateTensor(kv_list, num_layer_ * max_block_num * 2 * sizeof(void*));
  kv_list.dtype = TYPE_POINTER;

  CreateTensor(input_ids, max_token_num * sizeof(int));
  input_ids.dtype = TYPE_INT32;
  CreateTensor(input_offset_int32_tensor, (max_batch_size_ + 1) * sizeof(int));
  input_offset_int32_tensor.dtype = TYPE_INT32;
  CreateTensor(input_offset_uint64_tensor, (max_batch_size_ + 1) * sizeof(uint64_t));
  input_offset_uint64_tensor.dtype = TYPE_UINT64;
  CreateTensor(input_tokens_int32_tensor, (max_batch_size_ + 1) * sizeof(int));
  input_tokens_int32_tensor.dtype = TYPE_INT32;
  CreateTensor(rotary_embedding_pos, max_token_num * sizeof(int64_t));
  CreateTensor(forward_shape, sizeof(int));

  // 初始化各层实例
  emb_lookup_layer_ = std::make_shared<EmbLookupLayer>();
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  nccl_all_reduce_sum_layer_ = std::make_shared<NcclAllReduceSumLayer>();
  add_layer_ = std::make_shared<AddLayer>();
  silu_mul_layer_ = std::make_shared<SiluMulLayer>();
  matmul_layer_ = std::make_shared<MatMulLayer>();
  assemble_last_token_layer_ = std::make_shared<AssembleLastTokenLayer>();
  cast_layer_ = std::make_shared<CastLayer>();

  emb_lookup_layer_->Init({}, context_, rank_);
  layernorm_layer_->Init({layernorm_eps_}, context_, rank_);
  nccl_all_reduce_sum_layer_->Init({}, context_, rank_);
  add_layer_->Init({}, context_, rank_);
  silu_mul_layer_->Init({}, context_, rank_);
  matmul_layer_->Init({}, context_, rank_);
  assemble_last_token_layer_->Init({}, context_, rank_);
  cast_layer_->Init({}, context_, rank_);
  flash_attention_layer_.resize(num_layer_);
  paged_attention_layer_.resize(num_layer_);
  for (int idx = 0; idx < num_layer_; ++idx) {
    flash_attention_layer_[idx] = std::make_shared<FlashAttentionLayer>();
    paged_attention_layer_[idx] = std::make_shared<PagedAttentionLayer>();
    flash_attention_layer_[idx]->Init(
        {idx, max_position_embeddings, head_num, num_key_value_heads, size_per_head, stride_size, rotary_embedding,
         rope_theta, /*is_neox*/ true, std::any(cos_sin_cache_ptr)},
        context_, rank_);
    paged_attention_layer_[idx]->Init(
        {idx, max_position_embeddings, head_num, num_key_value_heads, size_per_head, stride_size, rotary_embedding,
         rope_theta, /*is_neox*/ true, std::any(cos_sin_cache_ptr)},
        context_, rank_);
  }

  // init cuda event
  CUDA_CHECK(cudaEventCreateWithFlags(&kvcache_offset_event_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&rotary_embedding_event_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&input_ids_event_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&nccl_finish_event_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&compute_ready_event_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&logits_transfer_event_, cudaEventDisableTiming));
}

template <typename T>
float* Llama<T>::GetLogitsPtr() {
  GetBlockManager()->SetDeviceId(rank_);
  return logits_tensor_.GetPtr<float>();
}

template <typename T>
void Llama<T>::PrepareKVCache(const size_t batch_size, size_t& total_seq_len, size_t& total_block_num,
                              const std::vector<ForwardRequest>& forward_reqs, std::vector<int>& kv_cache_offset_list,
                              cudaStream_t& stream, cudaEvent_t& event, bool context_stage) {
  for (size_t idx = 0; idx < batch_size; ++idx) {
    total_seq_len += forward_reqs[idx].output_tokens->size();
    total_block_num += forward_reqs[idx].kv_cache_ptrs[rank_].size();
    kv_cache_offset_list.push_back(total_block_num);
  }
  kv_cache_offset_tensor.shape = {kv_cache_offset_list.size()};
  void* kv_cache_offset_ptr = kv_cache_offset_tensor.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(kv_cache_offset_ptr, kv_cache_offset_list.data(),
                             kv_cache_offset_list.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
  NLLM_LOG_DEBUG << (context_stage ? "ContextDecode" : "Decode") << " Total Block Num " << total_block_num;
  kv_list.shape = {num_layer_, total_block_num * 2};
  std::vector<void*> cpu_kv_list(num_layer_ * total_block_num * 2);
  for (size_t layer_idx = 0; layer_idx < num_layer_; ++layer_idx) {
    int kv_list_index = 0;
    // 处理k
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = forward_reqs[idx].kv_cache_ptrs[rank_].size();
      for (size_t block_idx = 0; block_idx < block_num; block_idx++) {
        void* kv_cache_ptr = forward_reqs[idx].kv_cache_ptrs[rank_][block_idx];
        kv_cache_ptr += layer_idx * block_size_ / num_layer_;
        cpu_kv_list[layer_idx * total_block_num * 2 + kv_list_index] = kv_cache_ptr;
        kv_list_index++;
      }
    }
    // 处理v
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = forward_reqs[idx].kv_cache_ptrs[rank_].size();
      for (size_t block_idx = 0; block_idx < block_num; block_idx++) {
        void* kv_cache_ptr = forward_reqs[idx].kv_cache_ptrs[rank_][block_idx];
        kv_cache_ptr += layer_idx * block_size_ / num_layer_ + block_size_ / num_layer_ / 2;
        cpu_kv_list[layer_idx * total_block_num * 2 + kv_list_index] = kv_cache_ptr;
        kv_list_index++;
      }
    }
  }
  void* kv_list_ptr = kv_list.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(kv_list_ptr, cpu_kv_list.data(), cpu_kv_list.size() * sizeof(void*),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(event, stream));
}

template <typename T>
void Llama<T>::PrepareContextRotaryEmbeddingPos(const size_t batch_size, const size_t total_seq_len,
                                                const std::vector<ForwardRequest>& forward_reqs, cudaStream_t& stream,
                                                cudaEvent_t& event) {
  std::vector<int64_t> cpu_rotary_pos(total_seq_len);
  int cpu_rotary_pos_idx = 0;
  for (size_t idx = 0; idx < batch_size; ++idx) {
    for (size_t pos = 0; pos < forward_reqs[idx].output_tokens->size(); ++pos) {
      cpu_rotary_pos[cpu_rotary_pos_idx++] = pos;
    }
  }
  void* rotary_embedding_pos_ptr = rotary_embedding_pos.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(rotary_embedding_pos_ptr, cpu_rotary_pos.data(), sizeof(int64_t) * total_seq_len,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(event, stream));
}

template <typename T>
void Llama<T>::PrepareRotaryEmbeddingPos(const size_t batch_size, const std::vector<ForwardRequest>& forward_reqs,
                                         cudaStream_t& stream, cudaEvent_t& event) {
  std::vector<int64_t> cpu_rotary_pos(batch_size);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    cpu_rotary_pos[idx] = forward_reqs[idx].output_tokens->size() - 1;
  }
  void* rotary_embedding_pos_ptr = rotary_embedding_pos.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(rotary_embedding_pos_ptr, cpu_rotary_pos.data(), sizeof(int64_t) * batch_size,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(event, stream));
}

template <typename T>
void Llama<T>::PrepareContextInputIds(const size_t batch_size, const size_t total_seq_len, int& max_tokens,
                                      const std::vector<ForwardRequest>& forward_reqs, cudaStream_t& stream,
                                      cudaEvent_t& event) {
  input_ids.shape = {total_seq_len};
  int* input_ids_ptr = input_ids.GetPtr<int>();
  size_t input_offset = 0;
  std::vector<int> input_offset_list_int32(batch_size + 1, 0);
  std::vector<size_t> input_offset_list_uint64(batch_size + 1, 0ul);
  std::vector<int> input_ids_cpu(0);
  for (int idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t length = req_input->size();
    input_ids_cpu.insert(input_ids_cpu.end(), req_input->begin(), req_input->end());
    input_offset += length;
    input_offset_list_int32[idx + 1] = static_cast<int>(input_offset);
    input_offset_list_uint64[idx + 1] = input_offset;
    max_tokens = std::max(max_tokens, static_cast<int>(length));
  }
  CUDA_CHECK(cudaMemcpyAsync(input_ids_ptr, input_ids_cpu.data(), input_ids_cpu.size() * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  // create input offset tensor int32 and uint64
  input_offset_int32_tensor.shape = {batch_size + 1};
  input_offset_uint64_tensor.shape = {batch_size + 1};
  input_offset_int32_tensor.dtype = TYPE_INT32;
  input_offset_uint64_tensor.dtype = TYPE_UINT64;
  void* input_offset_int32_ptr = input_offset_int32_tensor.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(input_offset_int32_ptr, input_offset_list_int32.data(), (batch_size + 1) * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  void* input_offset_uint64_ptr = input_offset_uint64_tensor.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(input_offset_uint64_ptr, input_offset_list_uint64.data(),
                             (batch_size + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(event, stream));
}

template <typename T>
void Llama<T>::PrepareInputIds(const size_t batch_size, int& max_tokens,
                               const std::vector<ForwardRequest>& forward_reqs, cudaStream_t& stream,
                               cudaEvent_t& event) {
  input_ids.shape = {batch_size};
  void* input_ids_ptr = input_ids.GetPtr<void>();
  std::vector<int> input_ids_cpu(batch_size);
  size_t input_offset = 0;
  std::vector<int> input_offset_list_int32(batch_size + 1, 0);
  std::vector<int> input_tokens_list_int32(batch_size, 0);
  std::vector<size_t> input_offset_list_uint64(batch_size + 1, 0ul);
  for (int idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t length = req_input->size();
    input_ids_cpu[idx] = req_input->at(length - 1);
    max_tokens = std::max(max_tokens, int(length));
    input_offset++;
    input_tokens_list_int32[idx] = length;
    input_offset_list_int32[idx + 1] = static_cast<int>(input_offset);
    input_offset_list_uint64[idx + 1] = input_offset;
  }
  CUDA_CHECK(
      cudaMemcpyAsync(input_ids_ptr, input_ids_cpu.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
  // create input offset tensor int32 and uint64
  input_offset_int32_tensor.shape = {batch_size + 1};
  input_tokens_int32_tensor.shape = {batch_size};
  input_offset_uint64_tensor.shape = {batch_size + 1};
  void* input_offset_int32_ptr = input_offset_int32_tensor.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(input_offset_int32_ptr, input_offset_list_int32.data(), (batch_size + 1) * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  void* input_tokens_int32_ptr = input_tokens_int32_tensor.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(input_tokens_int32_ptr, input_tokens_list_int32.data(), (batch_size) * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  void* input_offset_uint64_ptr = input_offset_uint64_tensor.GetPtr<void>();
  CUDA_CHECK(cudaMemcpyAsync(input_offset_uint64_ptr, input_offset_list_uint64.data(),
                             (batch_size + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(event, stream));
}

template <typename T>
void Llama<T>::CopyToLogistBuffer(const size_t batch_size, cudaEvent_t& compute_ready_event_,
                                  cudaStream_t& compute_stream, cudaStream_t& d2d_stream,
                                  std::vector<ForwardRequest>& forward_reqs, std::vector<Tensor>& logits_float) {
  CUDA_CHECK(cudaEventRecord(compute_ready_event_, compute_stream));
  CUDA_CHECK(cudaStreamWaitEvent(d2d_stream, compute_ready_event_));
  // Copy to logits buf
  float* logits_ptr = logits_float[0].GetPtr<float>();
  float* logits_dst = forward_reqs[0].logits_buf[rank_] + forward_reqs[0].logits_offset * vocab_size_;
  CUDA_CHECK(cudaMemcpyAsync(logits_dst, logits_ptr, batch_size * vocab_size_ * sizeof(float), cudaMemcpyDeviceToDevice,
                             d2d_stream));
  CUDA_CHECK(cudaStreamSynchronize(d2d_stream));
}

template <typename T>
Status Llama<T>::ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                               std::vector<ForwardRequest>& forward_reqs) {
  GetBlockManager()->SetDeviceId(rank_);

  size_t batch_size = forward_reqs.size();
  NLLM_LOG_DEBUG << "ContextDecode With Batch Size " << batch_size;

  if (batch_size > max_batch_size_) {
    std::invalid_argument(
        fmt::format("Context Decode Batch Size out of max batch size! {} > {}", batch_size, max_batch_size_));
  }

  cudaStream_t& stream = context_->GetComputeStreams()[rank_];
  cudaStream_t& h2d_stream = context_->GetH2DStreams()[rank_];
  cudaStream_t& d2h_stream = context_->GetD2HStreams()[rank_];
  cudaStream_t& nccl_stream = context_->GetNCCLStreams()[rank_];
  cudaStream_t& d2d_stream = context_->GetD2HStreams()[rank_];

  // 推理前准备三块循环使用的推理时临时空间, 用于暂存各层输出结果
  std::vector<Tensor> output_0{tmp_tensor_0};
  std::vector<Tensor> output_1{tmp_tensor_1};
  std::vector<Tensor> output_2{tmp_tensor_2};
  // 解析外部 CPU 输入,拷贝到 GPU Tensor 中
  size_t total_seq_len = 0;
  size_t total_block_num = 0;
  int max_tokens = 0;
  std::vector<int> kv_cache_offset_list(1, 0);

  PrepareKVCache(batch_size, total_seq_len, total_block_num, forward_reqs, kv_cache_offset_list, d2h_stream,
                 kvcache_offset_event_, true);
  PrepareContextRotaryEmbeddingPos(batch_size, total_seq_len, forward_reqs, d2h_stream, rotary_embedding_event_);

  PrepareContextInputIds(batch_size, total_seq_len, max_tokens, forward_reqs, h2d_stream, input_ids_event_);

  // create forward shape tensor
  forward_shape.shape = {batch_size, max_tokens, kv_cache_offset_list.back()};

  // Forward
  // embedding
  Tensor embedding_weight = base_weight->GetModelWeights("gather_embedding");
  std::vector<Tensor>& emb_lookup_output = output_0;
  CUDA_CHECK(cudaStreamWaitEvent(stream, input_ids_event_));
  STATUS_CHECK_RETURN(
      emb_lookup_layer_->Forward({input_ids, input_offset_uint64_tensor, embedding_weight}, emb_lookup_output));
  emb_lookup_output[0].SaveToFile(saved_dir + "emb_lookup_output." + std::to_string(rank_) + ".npy");
  NLLM_LOG_DEBUG << "embeddig";

  // LlamaDecoder
  for (int layer_num = 0; layer_num < num_layer_; ++layer_num) {
    // input layernorm
    Tensor input_layernorm_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".input_layernorm");
    // input_layernorm_input = layer_num == 0 ? emb_lookup_output : mlp_add_output
    // Since emb_lookup_output and mlp_add_output point to the same memory address, we implement it as follow:
    std::vector<Tensor>& input_layernorm_input = output_0;
    std::vector<Tensor>& input_layernorm_output = output_1;
    STATUS_CHECK_RETURN(
        layernorm_layer_->Forward({input_layernorm_input[0], input_layernorm_weight}, input_layernorm_output));

    input_layernorm_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".input_layernorm." +
                                         std::to_string(rank_) + ".npy");
    NLLM_LOG_DEBUG << layer_num << " input layernorm";

    // Attn proj MatMul
    Tensor attn_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".attention.query_key_value");
    std::vector<Tensor>& attn_proj_output = output_2;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({input_layernorm_output[0], attn_proj_weight}, attn_proj_output));

    attn_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.proj." + std::to_string(rank_) +
                                   ".npy");
    NLLM_LOG_DEBUG << layer_num << " attn proj";

    // MMHA Flash Attention
    if (layer_num == 0) {
      // only need sync in the first layer
      CUDA_CHECK(cudaStreamWaitEvent(stream, kvcache_offset_event_));
      CUDA_CHECK(cudaStreamWaitEvent(stream, rotary_embedding_event_));
    }
    std::vector<Tensor>& flash_attention_output = output_1;
    STATUS_CHECK_RETURN(
        flash_attention_layer_[layer_num]->Forward({attn_proj_output[0], input_offset_uint64_tensor, kv_list,
                                                    kv_cache_offset_tensor, rotary_embedding_pos, forward_shape},
                                                   flash_attention_output));
    flash_attention_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.MMHA." +
                                         std::to_string(rank_) + ".npy");
    NLLM_LOG_DEBUG << layer_num << " MMHA Flash Attention";

    // Attn o_proj MatMul
    Tensor attn_o_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".attention.dense");
    std::vector<Tensor>& attn_o_proj_output = output_2;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({flash_attention_output[0], attn_o_proj_weight}, attn_o_proj_output));
    attn_o_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.o_proj." +
                                     std::to_string(rank_) + ".npy");

    // NOTE(karlluo): multiple event in nccl will cause preformance regression
    // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      CUDA_CHECK(cudaEventRecord(compute_ready_event_, stream));
      CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, compute_ready_event_));
    }
    // Attn NcclAllReduceSum
    std::vector<Tensor>& attn_all_reduce_sum_output = output_1;
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(attn_o_proj_output, attn_all_reduce_sum_output));
    attn_all_reduce_sum_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".attn_all_reduce_sum." +
                                             std::to_string(rank_) + ".npy");
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      CUDA_CHECK(cudaEventRecord(nccl_finish_event_, nccl_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, nccl_finish_event_));
    }

    // Attn Add
    std::vector<Tensor>& attn_add_output = output_2;
    STATUS_CHECK_RETURN(
        add_layer_->Forward({input_layernorm_input[0], attn_all_reduce_sum_output[0]}, attn_add_output));
    attn_add_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.add." + std::to_string(rank_) +
                                  ".npy");

    // post_attention_layernorm
    Tensor post_layernorm_weight =
        base_weight->GetModelWeights(std::to_string(layer_num) + ".post_attention_layernorm");
    std::vector<Tensor>& post_layernorm_output = output_1;
    STATUS_CHECK_RETURN(layernorm_layer_->Forward({attn_add_output[0], post_layernorm_weight}, post_layernorm_output));
    post_layernorm_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".post_attention_layernorm." +
                                        std::to_string(rank_) + ".npy");

    // Mlp gate_proj MatMul
    Tensor gate_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.gate_proj");
    std::vector<Tensor>& gate_matmul_output = output_0;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output[0], gate_proj_weight}, gate_matmul_output));
    gate_matmul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.gate_proj." + std::to_string(rank_) +
                                     ".npy");

    // Mlp up_proj MatMul 由于 gate_proj 与 up_proj 为并行关系,因此此处使用额外空间存储 matmul 结果
    Tensor up_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.up_proj");
    std::vector<Tensor> up_matmul_output = {up_matmul_tensor};
    STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output[0], up_proj_weight}, up_matmul_output));
    up_matmul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.up_proj." + std::to_string(rank_) +
                                   ".npy");

    std::vector<Tensor>& silu_mul_output = output_1;
    STATUS_CHECK_RETURN(silu_mul_layer_->Forward({gate_matmul_output[0], up_matmul_output[0]}, silu_mul_output));
    silu_mul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.silu." + std::to_string(rank_) +
                                  ".npy");

    // Mlp down_proj MatMul
    Tensor down_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.down_proj");
    std::vector<Tensor>& down_proj_output = output_0;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({silu_mul_output[0], down_proj_weight}, down_proj_output));
    down_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.down_proj." + std::to_string(rank_) +
                                   ".npy");

    // NOTE(karlluo): multiple event in nccl will cause preformance regression
    // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      CUDA_CHECK(cudaEventRecord(compute_ready_event_, stream));
      CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, compute_ready_event_));
    }
    // Mlp NcclAllReduceSum
    std::vector<Tensor>& mlp_all_reduce_sum_output = output_1;
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward({down_proj_output[0]}, mlp_all_reduce_sum_output));
    mlp_all_reduce_sum_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.nccl_all_reducesum." +
                                            std::to_string(rank_) + ".npy");
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      CUDA_CHECK(cudaEventRecord(nccl_finish_event_, nccl_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, nccl_finish_event_));
    }

    // Mlp Add
    std::vector<Tensor>& mlp_add_output = output_0;
    STATUS_CHECK_RETURN(add_layer_->Forward({mlp_all_reduce_sum_output[0], attn_add_output[0]}, mlp_add_output));
    mlp_add_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.add." + std::to_string(rank_) + ".npy");
  }

  // final norm
  Tensor final_layernorm_weight = base_weight->GetModelWeights("norm");
  std::vector<Tensor>& final_layernorm_input = output_0;
  std::vector<Tensor>& final_layernorm_output = output_1;
  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({final_layernorm_input[0], final_layernorm_weight}, final_layernorm_output));
  final_layernorm_input[0].SaveToFile(saved_dir + "final_norm.input." + std::to_string(rank_) + ".npy");
  final_layernorm_output[0].SaveToFile(saved_dir + "final_norm." + std::to_string(rank_) + ".npy");

  // assemble last token
  std::vector<Tensor>& assemble_last_token_output = output_2;
  STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward({final_layernorm_output[0], input_offset_uint64_tensor},
                                                          assemble_last_token_output));
  assemble_last_token_output[0].SaveToFile(saved_dir + "assemble_last_token." + std::to_string(rank_) + ".npy");

  // lm_head
  Tensor lm_head_weight = base_weight->GetModelWeights("lm_head");
  lm_head_weight.SaveToFile(saved_dir + "lm_head.weight." + std::to_string(rank_) + ".npy");
  std::vector<Tensor>& lm_head_output = output_0;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({assemble_last_token_output[0], lm_head_weight}, lm_head_output));
  lm_head_output[0].SaveToFile(saved_dir + "lm_head." + std::to_string(rank_) + ".npy");

  // Cast to float
  std::vector<Tensor>& logits_float = output_1;
  logits_float[0].dtype = TYPE_FP32;
  STATUS_CHECK_RETURN(cast_layer_->Forward(lm_head_output, logits_float));
  logits_float[0].SaveToFile(saved_dir + "logits_float." + std::to_string(rank_) + ".npy");

  CopyToLogistBuffer(batch_size, compute_ready_event_, stream, d2d_stream, forward_reqs, logits_float);
  return Status();
}

template <typename T>
Status Llama<T>::Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest>& forward_reqs) {
  GetBlockManager()->SetDeviceId(rank_);
  NLLM_LOG_DEBUG << "llama decode stage inference";

  saved_dir = "/model/llama-ft/7B/nllm_decode/";

  size_t batch_size = forward_reqs.size();
  NLLM_LOG_DEBUG << "Decode Batch_size = " << batch_size;
  if (batch_size > max_batch_size_) {
    std::invalid_argument(fmt::format("Decode Batch Size out of max batch size! {} > {}", batch_size, max_batch_size_));
  }

  cudaStream_t& stream = context_->GetComputeStreams()[rank_];
  cudaStream_t& h2d_stream = context_->GetH2DStreams()[rank_];
  cudaStream_t& d2h_stream = context_->GetD2HStreams()[rank_];
  cudaStream_t& nccl_stream = context_->GetNCCLStreams()[rank_];
  cudaStream_t& d2d_stream = context_->GetD2HStreams()[rank_];

  // 推理前准备三块循环使用的推理时临时空间, 用于暂存各层输出结果
  std::vector<Tensor> output_0{tmp_tensor_0};
  std::vector<Tensor> output_1{tmp_tensor_1};
  std::vector<Tensor> output_2{tmp_tensor_2};

  size_t total_seq_len = 0;
  size_t total_block_num = 0;
  int max_tokens = 0;
  std::vector<int> kv_cache_offset_list(1, 0);

  // prepare inputs
  PrepareKVCache(batch_size, total_seq_len, total_block_num, forward_reqs, kv_cache_offset_list, d2h_stream,
                 kvcache_offset_event_, false);
  PrepareRotaryEmbeddingPos(batch_size, forward_reqs, d2h_stream, rotary_embedding_event_);
  PrepareInputIds(batch_size, max_tokens, forward_reqs, h2d_stream, input_ids_event_);

  // create forward shape tensor
  forward_shape.shape = {batch_size, max_tokens, kv_cache_offset_list.back()};

  // Forward
  // embedding
  Tensor embedding_weight = base_weight->GetModelWeights("gather_embedding");
  std::vector<Tensor>& emb_lookup_output = output_0;
  CUDA_CHECK(cudaStreamWaitEvent(stream, input_ids_event_));
  STATUS_CHECK_RETURN(
      emb_lookup_layer_->Forward({input_ids, input_offset_uint64_tensor, embedding_weight}, emb_lookup_output));
  emb_lookup_output[0].SaveToFile(saved_dir + "emb_lookup_output." + std::to_string(rank_) + ".npy");

  // LlamaDecoder
  for (int layer_num = 0; layer_num < num_layer_; ++layer_num) {
    // input layernorm
    Tensor input_layernorm_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".input_layernorm");
    // input_layernorm_input = layer_num == 0 ? emb_lookup_output : mlp_add_output
    // Since emb_lookup_output and mlp_add_output point to the same memory address, we implement it as follow:
    std::vector<Tensor>& input_layernorm_input = output_0;
    std::vector<Tensor>& input_layernorm_output = output_1;
    STATUS_CHECK_RETURN(
        layernorm_layer_->Forward({input_layernorm_input[0], input_layernorm_weight}, input_layernorm_output));

    input_layernorm_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".input_layernorm." +
                                         std::to_string(rank_) + ".npy");

    // Attn proj MatMul
    Tensor attn_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".attention.query_key_value");
    std::vector<Tensor>& attn_proj_output = output_2;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({input_layernorm_output[0], attn_proj_weight}, attn_proj_output));

    attn_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.proj." + std::to_string(rank_) +
                                   ".npy");

    // MMHA Paged Attention
    std::vector<Tensor>& paged_attention_output = output_1;
    if (layer_num == 0) {
      // only need sync in the first layer
      CUDA_CHECK(cudaStreamWaitEvent(stream, kvcache_offset_event_));
      CUDA_CHECK(cudaStreamWaitEvent(stream, rotary_embedding_event_));
    }
    // TODO(zezhao): workspace 需与 catheywang 确认传入大小,暂时使用此处用不到的空间跑通流程
    STATUS_CHECK_RETURN(paged_attention_layer_[layer_num]->Forward(
        {attn_proj_output[0], input_tokens_int32_tensor, kv_list, kv_cache_offset_tensor, rotary_embedding_pos,
         kv_cache_buffer_, forward_shape, up_matmul_tensor},
        paged_attention_output));

    paged_attention_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.MMHA." +
                                         std::to_string(rank_) + ".npy");

    // Attn o_proj MatMul
    Tensor attn_o_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".attention.dense");
    std::vector<Tensor>& attn_o_proj_output = output_2;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({paged_attention_output[0], attn_o_proj_weight}, attn_o_proj_output));
    attn_o_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.o_proj." +
                                     std::to_string(rank_) + ".npy");

    // NOTE(karlluo): multiple event in nccl will cause preformance regression
    // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      CUDA_CHECK(cudaEventRecord(compute_ready_event_, stream));
      CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, compute_ready_event_));
    }
    // Attn NcclAllReduceSum
    std::vector<Tensor>& attn_all_reduce_sum_output = output_1;
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(attn_o_proj_output, attn_all_reduce_sum_output));
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      CUDA_CHECK(cudaEventRecord(nccl_finish_event_, nccl_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, nccl_finish_event_));
    }

    // Attn Add
    std::vector<Tensor>& attn_add_output = output_2;
    STATUS_CHECK_RETURN(
        add_layer_->Forward({input_layernorm_input[0], attn_all_reduce_sum_output[0]}, attn_add_output));
    attn_add_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.add." + std::to_string(rank_) +
                                  ".npy");

    // post_attention_layernorm
    Tensor post_layernorm_weight =
        base_weight->GetModelWeights(std::to_string(layer_num) + ".post_attention_layernorm");
    std::vector<Tensor>& post_layernorm_output = output_1;
    STATUS_CHECK_RETURN(layernorm_layer_->Forward({attn_add_output[0], post_layernorm_weight}, post_layernorm_output));
    post_layernorm_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".post_attention_layernorm." +
                                        std::to_string(rank_) + ".npy");

    // Mlp gate_proj MatMul
    Tensor gate_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.gate_proj");
    std::vector<Tensor>& gate_matmul_output = output_0;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output[0], gate_proj_weight}, gate_matmul_output));
    gate_matmul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.gate_proj." + std::to_string(rank_) +
                                     ".npy");

    // Mlp up_proj MatMul 由于 gate_proj 与 up_proj 为并行关系,因此此处使用额外空间存储 matmul 结果
    Tensor up_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.up_proj");
    std::vector<Tensor> up_matmul_output = {up_matmul_tensor};
    STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output[0], up_proj_weight}, up_matmul_output));
    up_matmul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.up_proj." + std::to_string(rank_) +
                                   ".npy");

    std::vector<Tensor>& silu_mul_output = output_1;
    STATUS_CHECK_RETURN(silu_mul_layer_->Forward({gate_matmul_output[0], up_matmul_output[0]}, silu_mul_output));
    silu_mul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.silu." + std::to_string(rank_) +
                                  ".npy");

    // Mlp down_proj MatMul
    Tensor down_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.down_proj");
    std::vector<Tensor>& down_proj_output = output_0;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({silu_mul_output[0], down_proj_weight}, down_proj_output));
    down_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.down_proj." + std::to_string(rank_) +
                                   ".npy");

    // NOTE(karlluo): multiple event in nccl will cause preformance regression
    // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      CUDA_CHECK(cudaEventRecord(compute_ready_event_, stream));
      CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, compute_ready_event_));
    }
    // Mlp NcclAllReduceSum
    std::vector<Tensor>& mlp_all_reduce_sum_output = output_1;
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward({down_proj_output[0]}, mlp_all_reduce_sum_output));
    mlp_all_reduce_sum_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.nccl_all_reducesum." +
                                            std::to_string(rank_) + ".npy");
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      CUDA_CHECK(cudaEventRecord(nccl_finish_event_, nccl_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, nccl_finish_event_));
    }

    // Mlp Add
    std::vector<Tensor>& mlp_add_output = output_0;
    STATUS_CHECK_RETURN(add_layer_->Forward({mlp_all_reduce_sum_output[0], attn_add_output[0]}, mlp_add_output));
    mlp_add_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.add." + std::to_string(rank_) + ".npy");
  }

  // final norm
  Tensor final_layernorm_weight = base_weight->GetModelWeights("norm");
  std::vector<Tensor>& final_layernorm_input = output_0;
  std::vector<Tensor>& final_layernorm_output = output_1;
  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({final_layernorm_input[0], final_layernorm_weight}, final_layernorm_output));
  final_layernorm_output[0].SaveToFile(saved_dir + "final_norm." + std::to_string(rank_) + ".npy");

  // assemble last token
  std::vector<Tensor>& assemble_last_token_output = output_2;
  STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward({final_layernorm_output[0], input_offset_uint64_tensor},
                                                          assemble_last_token_output));

  assemble_last_token_output[0].SaveToFile(saved_dir + "assemble_last_token." + std::to_string(rank_) + ".npy");

  // lm_head
  Tensor lm_head_weight = base_weight->GetModelWeights("lm_head");
  lm_head_weight.SaveToFile(saved_dir + "lm_head.weight." + std::to_string(rank_) + ".npy");
  std::vector<Tensor>& lm_head_output = output_0;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({assemble_last_token_output[0], lm_head_weight}, lm_head_output));
  lm_head_output[0].SaveToFile(saved_dir + "lm_head." + std::to_string(rank_) + ".npy");

  // Cast to float
  std::vector<Tensor>& logits_float = output_1;
  logits_float[0].dtype = TYPE_FP32;
  STATUS_CHECK_RETURN(cast_layer_->Forward(lm_head_output, logits_float));
  logits_float[0].SaveToFile(saved_dir + "logits_float." + std::to_string(rank_) + ".npy");

  CopyToLogistBuffer(batch_size, compute_ready_event_, stream, d2d_stream, forward_reqs, logits_float);
  return Status();
}

template class Llama<float>;
template class Llama<half>;

}  // namespace ksana_llm
