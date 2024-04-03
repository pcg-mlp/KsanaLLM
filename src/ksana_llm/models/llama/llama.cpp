/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/llama/llama.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
Llama<T>::~Llama() {
  // free all cude event
  EventDestroy(kvcache_offset_event_);
  EventDestroy(rotary_embedding_event_);
  EventDestroy(input_ids_event_);
  EventDestroy(nccl_finish_event_);
  EventDestroy(compute_ready_event_);
  EventDestroy(logits_transfer_event_);
}

template <typename T>
Llama<T>::Llama(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context) {
  GetBlockManager()->SetDeviceId(rank_);

#ifdef ENABLE_CUDA
  context_ = context;
  rank_ = rank;
  num_layer_ = model_config.num_layer;
  weight_data_type_ = model_config.weight_data_type;
  vocab_size_ = model_config.vocab_size;

  float layernorm_eps_ = model_config.layernorm_eps;
  int head_num = model_config.head_num;
  int size_per_head = model_config.size_per_head;
  int hidden_units = size_per_head * head_num;
  int rotary_embedding = model_config.rotary_embedding;
  int head_num_per_tp = head_num / model_config.tensor_para_size;
  int num_key_value_heads = model_config.num_key_value_heads / model_config.tensor_para_size;
  int stride_size = head_num_per_tp * size_per_head * 3;
  int inter_size = model_config.inter_size;
  int max_position_embeddings = model_config.max_position_embeddings;
  float rope_theta = model_config.rope_theta;
  block_token_num_ = GetBlockManager()->GetBlockTokenNum();
  block_size_ = GetBlockManager()->GetBlockSize();

  BlockManagerConfig block_manager_config;
  STATUS_CHECK_FAILURE(Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config));

  int prefix_cache_tokens_number =
      block_manager_config.prefix_cache_len > 0 ? block_manager_config.prefix_cache_len : 0;

  max_batch_size_ = model_config.max_batch_size;
  max_seq_len_ = model_config.max_token_num;
  size_t dtype_size = GetTypeSize(weight_data_type_);
  size_t max_token_num = model_config.max_scheduler_token_num;
  max_token_num_ = max_token_num;
  qkv_add_bias_ = (model_config.type.find("qwen") != std::string::npos);
  NLLM_LOG_DEBUG << fmt::format("Max_Batch_Size = {}, Max Seq Len = {}, Max Token Num = {}", max_batch_size_,
                                max_seq_len_, max_token_num);

  size_t tensor_buffer_1_size =
      std::max(max_batch_size_ * vocab_size_ * sizeof(float), max_token_num * hidden_units * 3 * dtype_size) /
      GetTypeSize(weight_data_type_);
  size_t up_matmul_tensor_buffer_size = max_token_num * std::max(inter_size, hidden_units * 2);
  size_t max_block_num =
      (max_seq_len_ * max_batch_size_ + model_config.block_token_num - 1) / model_config.block_token_num;
  // NOTE(karlluo): when using prefixed cache, more tokens come into forward, hence we need more buffer.
  size_t extra_token_number = 0;
  if (prefix_cache_tokens_number > 0) {
    extra_token_number = prefix_cache_tokens_number * max_batch_size_;
  }

  // TODO(karlluo): all create tensor used dynamic memory pool
  STATUS_CHECK_FAILURE(CreateBufferTensor(tensor_buffer_0_, {max_token_num, hidden_units, 3}, weight_data_type_));
  STATUS_CHECK_FAILURE(CreateBufferTensor(tensor_buffer_1_, {tensor_buffer_1_size}, weight_data_type_))
  STATUS_CHECK_FAILURE(CreateBufferTensor(tensor_buffer_2_, {max_token_num, hidden_units, 3}, weight_data_type_));
  STATUS_CHECK_FAILURE(CreateBufferTensor(up_matmul_tensor_buffer_, {up_matmul_tensor_buffer_size}, weight_data_type_));
  STATUS_CHECK_FAILURE(CreateBufferTensor(
      kv_cache_buffer_, {max_seq_len_, (max_seq_len_ + 511) / 512, head_num_per_tp, size_per_head + 2}, TYPE_FP32));
  STATUS_CHECK_FAILURE(CreateBufferTensor(logits_tensor_, {max_batch_size_, vocab_size_}, TYPE_FP32));
  STATUS_CHECK_FAILURE(CreateBufferTensor(kv_cache_offset_tensor_, {max_batch_size_ + 1}, TYPE_INT32));
  STATUS_CHECK_FAILURE(
      CreateBufferTensor(cos_sin_cache_tensor_, {rotary_embedding, max_position_embeddings}, TYPE_FP16));
  STATUS_CHECK_FAILURE(CreateBufferTensor(kv_list_, {num_layer_, max_block_num, 2}, TYPE_POINTER));
  STATUS_CHECK_FAILURE(CreateBufferTensor(input_ids_, {max_token_num + extra_token_number}, TYPE_INT32));
  STATUS_CHECK_FAILURE(CreateBufferTensor(input_offset_int32_tensor_, {max_batch_size_ + 1}, TYPE_INT32));
  STATUS_CHECK_FAILURE(CreateBufferTensor(input_offset_uint64_tensor_, {max_batch_size_ + 1}, TYPE_UINT64));
  STATUS_CHECK_FAILURE(CreateBufferTensor(input_tokens_int32_tensor_, {max_batch_size_ + 1}, TYPE_INT32));
  STATUS_CHECK_FAILURE(CreateBufferTensor(rotary_embedding_pos_, {max_token_num + extra_token_number}, TYPE_INT64));
  // TODO(karlluo): we needn't tensor's shape to transfer attribute
  STATUS_CHECK_FAILURE(CreateBufferTensor(forward_shape_, {1}, TYPE_INT32));

  Event create_reduce_tensor_event;
  if (use_custom_all_reduce_) {
    EventCreateWithFlags(&create_reduce_tensor_event, EVENT_DISABLE_TIMING);
    // create buffer for custom all reduce sum
    constexpr size_t reduce_buffer_size = 256;
    STATUS_CHECK_FAILURE(CreateBufferTensor(reduce_tensor_, {reduce_buffer_size}, TYPE_UINT8));
    size_t rank_data_sz = context_->GetTensorParallelSize() * 128;
    STATUS_CHECK_FAILURE(CreateBufferTensor(rank_tensor_0_, {rank_data_sz}, TYPE_UINT8));
    EventRecord(create_reduce_tensor_event, context_->GetMemoryManageStreams()[rank_]);
  }

  NLLM_LOG_DEBUG << "Total buffer tensors memory used: " << (GetBufferTensorsMemoryUsed() >> 20) << " MB";

  // 初始化各层实例
  emb_lookup_layer_ = std::make_shared<EmbLookupLayer>();
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  if (use_custom_all_reduce_) {
    custom_all_reduce_sum_layer_0_ = std::make_shared<CustomAllReduceSumLayer>();
  }
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
  flash_attention_layers_.resize(num_layer_);
  paged_attention_layers_.resize(num_layer_);
  half* cos_sin_cache_ptr = cos_sin_cache_tensor_.GetPtr<half>();
  for (int idx = 0; idx < num_layer_; ++idx) {
    flash_attention_layers_[idx] = std::make_shared<FlashAttentionLayer>();
    paged_attention_layers_[idx] = std::make_shared<PagedAttentionLayer>();
    flash_attention_layers_[idx]->Init({idx, max_position_embeddings, head_num_per_tp, num_key_value_heads,
                                        size_per_head, stride_size, rotary_embedding, rope_theta, /*is_neox*/ true,
                                        std::any(cos_sin_cache_ptr), model_config.rope_scaling_factor_config},
                                       context_, rank_);
    paged_attention_layers_[idx]->Init({idx, max_position_embeddings, head_num_per_tp, num_key_value_heads,
                                        size_per_head, stride_size, rotary_embedding, rope_theta, /*is_neox*/ true,
                                        std::any(cos_sin_cache_ptr), model_config.rope_scaling_factor_config},
                                       context_, rank_);
  }

  if (use_custom_all_reduce_) {
    StreamWaitEvent(context_->GetMemoryManageStreams()[rank_], create_reduce_tensor_event);
    size_t reduce_buffer_size = 256;

    MemsetAsync(reduce_tensor_.GetPtr<void>(), 0, reduce_buffer_size, context_->GetMemoryManageStreams()[rank_]);

    size_t rank_data_sz = context_->GetTensorParallelSize() * 128;
    custom_all_reduce_sum_layer_0_->Init(
        {reduce_tensor_.GetPtr<void>(), tensor_buffer_0_.GetPtr<void>(), reduce_buffer_size,
         rank_tensor_0_.GetPtr<void>(), rank_data_sz, tensor_buffer_2_.GetPtr<void>(), 0},
        context_, rank_);
    EventDestroy(create_reduce_tensor_event);
  }

  // init cuda event
  EventCreateWithFlags(&kvcache_offset_event_, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&rotary_embedding_event_, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&input_ids_event_, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&nccl_finish_event_, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&compute_ready_event_, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&logits_transfer_event_, EVENT_DISABLE_TIMING);
#endif
}

template <typename T>
float* Llama<T>::GetLogitsPtr() {
  GetBlockManager()->SetDeviceId(rank_);
  return logits_tensor_.GetPtr<float>();
}

template <typename T>
void Llama<T>::PrepareKVCache(const size_t batch_size, size_t& total_seq_len, size_t& total_block_num,
                              const std::vector<ForwardRequest>& forward_reqs, std::vector<int>& kv_cache_offset_list,
                              Stream& stream, Event& event, bool is_context_stage) {
#ifdef ENABLE_CUDA
  for (size_t idx = 0; idx < batch_size; ++idx) {
    total_seq_len += forward_reqs[idx].output_tokens->size();
    total_block_num += forward_reqs[idx].kv_cache_ptrs[rank_].size();
    kv_cache_offset_list.push_back(total_block_num);
  }
  kv_cache_offset_tensor_.shape = {kv_cache_offset_list.size()};
  void* kv_cache_offset_ptr = kv_cache_offset_tensor_.GetPtr<void>();
  MemcpyAsync(kv_cache_offset_ptr, kv_cache_offset_list.data(), kv_cache_offset_list.size() * sizeof(int),
              MEMCPY_HOST_TO_DEVICE, stream);
  NLLM_LOG_DEBUG << (is_context_stage ? "ContextDecode" : "Decode") << " Total Block Num " << total_block_num;
  kv_list_.shape = {num_layer_, total_block_num * 2};
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
  void* kv_list_ptr = kv_list_.GetPtr<void>();
  MemcpyAsync(kv_list_ptr, cpu_kv_list.data(), cpu_kv_list.size() * sizeof(void*), MEMCPY_HOST_TO_DEVICE, stream);
  EventRecord(event, stream);
#endif
}

template <typename T>
void Llama<T>::PrepareContextRotaryEmbeddingPos(const size_t batch_size, const size_t total_seq_len,
                                                const std::vector<ForwardRequest>& forward_reqs, Stream& stream,
                                                Event& event) {
#ifdef ENABLE_CUDA
  std::vector<int64_t> cpu_rotary_pos(total_seq_len);
  int cpu_rotary_pos_idx = 0;
  for (size_t idx = 0; idx < batch_size; ++idx) {
    for (size_t pos = 0; pos < forward_reqs[idx].output_tokens->size(); ++pos) {
      cpu_rotary_pos[cpu_rotary_pos_idx++] = pos;
    }
  }
  void* rotary_embedding_pos_ptr = rotary_embedding_pos_.GetPtr<void>();
  MemcpyAsync(rotary_embedding_pos_ptr, cpu_rotary_pos.data(), sizeof(int64_t) * total_seq_len, MEMCPY_HOST_TO_DEVICE,
              stream);
  EventRecord(event, stream);
#endif
}

template <typename T>
void Llama<T>::PrepareRotaryEmbeddingPos(const size_t batch_size, const std::vector<ForwardRequest>& forward_reqs,
                                         Stream& stream, Event& event) {
#ifdef ENABLE_CUDA
  std::vector<int64_t> cpu_rotary_pos(batch_size);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    cpu_rotary_pos[idx] = forward_reqs[idx].output_tokens->size() - 1;
  }
  void* rotary_embedding_pos_ptr = rotary_embedding_pos_.GetPtr<void>();
  MemcpyAsync(rotary_embedding_pos_ptr, cpu_rotary_pos.data(), sizeof(int64_t) * batch_size, MEMCPY_HOST_TO_DEVICE,
              stream);
  EventRecord(event, stream);
#endif
}

template <typename T>
void Llama<T>::PrepareContextInputIds(const size_t batch_size, const size_t total_seq_len, int& max_tokens,
                                      const std::vector<ForwardRequest>& forward_reqs, Stream& stream, Event& event) {
#ifdef ENABLE_CUDA
  input_ids_.shape = {total_seq_len};
  int* input_ids_ptr = input_ids_.GetPtr<int>();
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
  MemcpyAsync(input_ids_ptr, input_ids_cpu.data(), input_ids_cpu.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE, stream);
  // create input offset tensor int32 and uint64
  input_offset_int32_tensor_.shape = {batch_size + 1};
  input_offset_uint64_tensor_.shape = {batch_size + 1};
  input_offset_int32_tensor_.dtype = TYPE_INT32;
  input_offset_uint64_tensor_.dtype = TYPE_UINT64;
  void* input_offset_int32_ptr = input_offset_int32_tensor_.GetPtr<void>();
  MemcpyAsync(input_offset_int32_ptr, input_offset_list_int32.data(), (batch_size + 1) * sizeof(int),
              MEMCPY_HOST_TO_DEVICE, stream);
  void* input_offset_uint64_ptr = input_offset_uint64_tensor_.GetPtr<void>();
  MemcpyAsync(input_offset_uint64_ptr, input_offset_list_uint64.data(), (batch_size + 1) * sizeof(size_t),
              MEMCPY_HOST_TO_DEVICE, stream);
  EventRecord(event, stream);
#endif
}

template <typename T>
void Llama<T>::PrepareInputIds(const size_t batch_size, int& max_tokens,
                               const std::vector<ForwardRequest>& forward_reqs, Stream& stream, Event& event) {
#ifdef ENABLE_CUDA
  input_ids_.shape = {batch_size};
  void* input_ids_ptr = input_ids_.GetPtr<void>();
  std::vector<int> input_ids_cpu(batch_size);
  size_t input_offset = 0;
  std::vector<int> input_offset_list_int32(batch_size + 1, 0);
  std::vector<int> input_tokens_list_int32(batch_size, 0);
  std::vector<size_t> input_offset_list_uint64(batch_size + 1, 0ul);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t length = req_input->size();
    input_ids_cpu[idx] = req_input->at(length - 1);
    max_tokens = std::max(max_tokens, int(length));
    input_offset++;
    input_tokens_list_int32[idx] = length;
    input_offset_list_int32[idx + 1] = static_cast<int>(input_offset);
    input_offset_list_uint64[idx + 1] = input_offset;
  }

  MemcpyAsync(input_ids_ptr, input_ids_cpu.data(), batch_size * sizeof(int), MEMCPY_HOST_TO_DEVICE, stream);
  // create input offset tensor int32 and uint64
  input_offset_int32_tensor_.shape = {batch_size + 1};
  input_tokens_int32_tensor_.shape = {batch_size};
  input_offset_uint64_tensor_.shape = {batch_size + 1};
  void* input_offset_int32_ptr = input_offset_int32_tensor_.GetPtr<void>();
  MemcpyAsync(input_offset_int32_ptr, input_offset_list_int32.data(), (batch_size + 1) * sizeof(int),
              MEMCPY_HOST_TO_DEVICE, stream);
  void* input_tokens_int32_ptr = input_tokens_int32_tensor_.GetPtr<void>();
  MemcpyAsync(input_tokens_int32_ptr, input_tokens_list_int32.data(), (batch_size) * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              stream);
  void* input_offset_uint64_ptr = input_offset_uint64_tensor_.GetPtr<void>();
  MemcpyAsync(input_offset_uint64_ptr, input_offset_list_uint64.data(), (batch_size + 1) * sizeof(size_t),
              MEMCPY_HOST_TO_DEVICE, stream);
  EventRecord(event, stream);
#endif
}

template <typename T>
void Llama<T>::CopyToLogistBuffer(const size_t batch_size, std::vector<ForwardRequest>& forward_reqs,
                                  std::vector<Tensor>& logits_float) {
#ifdef ENABLE_CUDA
  EventRecord(compute_ready_event_, context_->GetComputeStreams()[rank_]);
  StreamWaitEvent(context_->GetD2DStreams()[rank_], compute_ready_event_);
  // Copy to logits buf
  float* logits_ptr = logits_float[0].GetPtr<float>();
  float* logits_dst = forward_reqs[0].logits_buf[rank_] + forward_reqs[0].logits_offset * vocab_size_;
  MemcpyAsync(logits_dst, logits_ptr, batch_size * vocab_size_ * sizeof(float), MEMCPY_DEVICE_TO_DEVICE,
              context_->GetD2DStreams()[rank_]);
  StreamSynchronize(context_->GetD2DStreams()[rank_]);
#endif
}

template <typename T>
Status Llama<T>::LlamaAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                Tensor& hidden_states, std::vector<Tensor>& temp_buffer_0,
                                std::vector<Tensor>& temp_buffer_1, std::vector<Tensor>& temp_buffer_2,
                                const bool is_context_stage) {
#ifdef ENABLE_CUDA
  // Attn proj MatMul
  Tensor attn_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.query_key_value.weight", layer_idx));
  std::vector<Tensor>& attn_proj_output = temp_buffer_2;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({hidden_states, attn_proj_weight}, attn_proj_output));
  if (qkv_add_bias_) {
    Tensor attn_proj_bias =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.query_key_value.bias", layer_idx));
    STATUS_CHECK_RETURN(add_layer_->Forward({attn_proj_output[0], attn_proj_bias}, attn_proj_output));
  }

  // MMHA Flash/Paged Attention
  std::vector<Tensor>& mmha_attention_output = temp_buffer_1;
  if (layer_idx == 0) {
    // only need sync in the first layer
    StreamWaitEvent(context_->GetComputeStreams()[rank_], kvcache_offset_event_);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], rotary_embedding_event_);
  }

  if (is_context_stage) {
    STATUS_CHECK_RETURN(
        flash_attention_layers_[layer_idx]->Forward({attn_proj_output[0], input_offset_uint64_tensor_, kv_list_,
                                                     kv_cache_offset_tensor_, rotary_embedding_pos_, forward_shape_},
                                                    mmha_attention_output));
  } else {
    STATUS_CHECK_RETURN(paged_attention_layers_[layer_idx]->Forward(
        {attn_proj_output[0], input_tokens_int32_tensor_, kv_list_, kv_cache_offset_tensor_, rotary_embedding_pos_,
         kv_cache_buffer_, forward_shape_, up_matmul_tensor_buffer_},
        mmha_attention_output));
  }

  // Attn o_proj MatMul
  Tensor attn_o_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx));
  std::vector<Tensor>& attn_o_proj_output = temp_buffer_2;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({mmha_attention_output[0], attn_o_proj_weight}, attn_o_proj_output));

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(compute_ready_event_, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], compute_ready_event_);
  }
  // Attn NcclAllReduceSum
  std::vector<Tensor>& attn_all_reduce_sum_output = temp_buffer_1;
  if (is_context_stage) {
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(attn_o_proj_output, attn_all_reduce_sum_output));
  } else {
    if (use_custom_all_reduce_) {
      STATUS_CHECK_RETURN(custom_all_reduce_sum_layer_0_->Forward({attn_o_proj_output[0]}, attn_all_reduce_sum_output));
    } else {
      STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(attn_o_proj_output, attn_all_reduce_sum_output));
    }
  }
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(nccl_finish_event_, context_->GetNCCLStreams()[rank_]);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], nccl_finish_event_);
  }
#endif

  return Status();
}

template <typename T>
Status Llama<T>::LlamaMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                          Tensor& post_layernorm_output, std::vector<Tensor>& temp_buffer_0,
                          std::vector<Tensor>& temp_buffer_1, std::vector<Tensor>& temp_buffer_2) {
#ifdef ENABLE_CUDA
  // Mlp gate_proj MatMul
  Tensor gate_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.gate_proj.weight", layer_idx));
  std::vector<Tensor>& gate_matmul_output = temp_buffer_0;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output, gate_proj_weight}, gate_matmul_output));

  // Mlp up_proj MatMul 由于 gate_proj 与 up_proj 为并行关系,因此此处使用额外空间存储 matmul 结果
  Tensor up_proj_weight = base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.up_proj.weight", layer_idx));
  std::vector<Tensor> up_matmul_output = {up_matmul_tensor_buffer_};
  STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output, up_proj_weight}, up_matmul_output));

  std::vector<Tensor>& silu_mul_output = temp_buffer_1;
  STATUS_CHECK_RETURN(silu_mul_layer_->Forward({gate_matmul_output[0], up_matmul_output[0]}, silu_mul_output));

  // Mlp down_proj MatMul
  Tensor down_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.down_proj.weight", layer_idx));
  std::vector<Tensor>& down_proj_output = temp_buffer_0;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({silu_mul_output[0], down_proj_weight}, down_proj_output));

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(compute_ready_event_, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], compute_ready_event_);
  }
  // Mlp NcclAllReduceSum
  std::vector<Tensor>& mlp_all_reduce_sum_output = temp_buffer_1;
  STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward({down_proj_output[0]}, mlp_all_reduce_sum_output));
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(nccl_finish_event_, context_->GetNCCLStreams()[rank_]);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], nccl_finish_event_);
  }
#endif

  return Status();
}

template <typename T>
Status Llama<T>::LlamaDecoder(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                              std::vector<Tensor>& temp_buffer_0, std::vector<Tensor>& temp_buffer_1,
                              std::vector<Tensor>& temp_buffer_2, const bool is_context_stage) {
#ifdef ENABLE_CUDA
  // input layernorm
  Tensor input_layernorm_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.input_layernorm.weight", layer_idx));
  // input_layernorm_input = layer_idx == 0 ? emb_lookup_output : mlp_add_output
  // Since emb_lookup_output and mlp_add_output point to the same memory address, we implement it as follow:
  std::vector<Tensor>& input_layernorm_input = temp_buffer_0;
  std::vector<Tensor>& input_layernorm_output = temp_buffer_1;
  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({input_layernorm_input[0], input_layernorm_weight}, input_layernorm_output));

  STATUS_CHECK_RETURN(LlamaAttention(layer_idx, base_weight, input_layernorm_output[0], temp_buffer_0, temp_buffer_1,
                                     temp_buffer_2,
                                     /*is_context_stage*/ is_context_stage));

  // Attn Add
  std::vector<Tensor>& attn_all_reduce_sum_output = temp_buffer_1;
  std::vector<Tensor>& attn_add_output = temp_buffer_2;
  STATUS_CHECK_RETURN(add_layer_->Forward({input_layernorm_input[0], attn_all_reduce_sum_output[0]}, attn_add_output));

  // post_attention_layernorm
  Tensor post_layernorm_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.post_attention_layernorm.weight", layer_idx));
  std::vector<Tensor>& post_layernorm_output = temp_buffer_1;
  STATUS_CHECK_RETURN(layernorm_layer_->Forward({attn_add_output[0], post_layernorm_weight}, post_layernorm_output));

  STATUS_CHECK_RETURN(
      LlamaMlp(layer_idx, base_weight, post_layernorm_output[0], temp_buffer_0, temp_buffer_1, temp_buffer_2));

  // Mlp Add
  std::vector<Tensor>& mlp_all_reduce_sum_output = temp_buffer_1;
  std::vector<Tensor>& mlp_add_output = temp_buffer_0;
  STATUS_CHECK_RETURN(add_layer_->Forward({mlp_all_reduce_sum_output[0], attn_add_output[0]}, mlp_add_output));
#endif
  return Status();
}

template <typename T>
Status Llama<T>::ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                               std::vector<ForwardRequest>& forward_reqs) {
  GetBlockManager()->SetDeviceId(rank_);
#ifdef ENABLE_CUDA

  size_t batch_size = forward_reqs.size();
  NLLM_LOG_DEBUG << "ContextDecode With Batch Size " << batch_size;
  if (batch_size > max_batch_size_) {
    std::invalid_argument(
        fmt::format("Context Decode Batch Size out of max batch size! {} > {}", batch_size, max_batch_size_));
  }

  // 推理前准备三块循环使用的推理时临时空间, 用于暂存各层输出结果
  std::vector<Tensor> temp_buffer_0{tensor_buffer_0_};
  std::vector<Tensor> temp_buffer_1{tensor_buffer_1_};
  std::vector<Tensor> temp_buffer_2{tensor_buffer_2_};
  // 解析外部 CPU 输入,拷贝到 GPU Tensor 中
  size_t total_seq_len = 0;
  size_t total_block_num = 0;
  int max_tokens = 0;
  std::vector<int> kv_cache_offset_list(1, 0);

  // prepare all inputs
  PrepareKVCache(batch_size, total_seq_len, total_block_num, forward_reqs, kv_cache_offset_list,
                 context_->GetD2HStreams()[rank_], kvcache_offset_event_, true);
  PrepareContextRotaryEmbeddingPos(batch_size, total_seq_len, forward_reqs, context_->GetD2HStreams()[rank_],
                                   rotary_embedding_event_);
  PrepareContextInputIds(batch_size, total_seq_len, max_tokens, forward_reqs, context_->GetH2DStreams()[rank_],
                         input_ids_event_);

  // create forward shape tensor
  forward_shape_.shape = {batch_size, max_tokens, kv_cache_offset_list.back()};

  // embedding
  Tensor embedding_weight = base_weight->GetModelWeights("model.embed_tokens.weight");
  std::vector<Tensor>& emb_lookup_output = temp_buffer_0;
  StreamWaitEvent(context_->GetComputeStreams()[rank_], input_ids_event_);
  STATUS_CHECK_RETURN(
      emb_lookup_layer_->Forward({input_ids_, input_offset_uint64_tensor_, embedding_weight}, emb_lookup_output));

  // LlamaDecoder
  for (int layer_idx = 0; layer_idx < num_layer_; ++layer_idx) {
    STATUS_CHECK_RETURN(LlamaDecoder(layer_idx, base_weight, temp_buffer_0, temp_buffer_1, temp_buffer_2,
                                     /*is_context_stage*/ true));
  }

  // final norm
  Tensor final_layernorm_weight = base_weight->GetModelWeights("model.norm.weight");
  std::vector<Tensor>& final_layernorm_input = temp_buffer_0;
  std::vector<Tensor>& final_layernorm_output = temp_buffer_1;
  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({final_layernorm_input[0], final_layernorm_weight}, final_layernorm_output));

  // assemble last token
  std::vector<Tensor>& assemble_last_token_output = temp_buffer_2;
  STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward({final_layernorm_output[0], input_offset_uint64_tensor_},
                                                          assemble_last_token_output));

  // lm_head
  Tensor lm_head_weight = base_weight->GetModelWeights("lm_head.weight");
  std::vector<Tensor>& lm_head_output = temp_buffer_0;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({assemble_last_token_output[0], lm_head_weight}, lm_head_output));

  // Cast to float
  std::vector<Tensor>& logits_float = temp_buffer_1;
  logits_float[0].dtype = TYPE_FP32;
  STATUS_CHECK_RETURN(cast_layer_->Forward(lm_head_output, logits_float));

  CopyToLogistBuffer(batch_size, forward_reqs, logits_float);
#endif
  return Status();
}

template <typename T>
Status Llama<T>::Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest>& forward_reqs) {
  GetBlockManager()->SetDeviceId(rank_);
#ifdef ENABLE_CUDA
  size_t batch_size = forward_reqs.size();
  NLLM_LOG_DEBUG << "Decode Batch_size = " << batch_size;
  if (batch_size > max_batch_size_) {
    std::invalid_argument(fmt::format("Decode Batch Size out of max batch size! {} > {}", batch_size, max_batch_size_));
  }

  // 推理前准备三块循环使用的推理时临时空间, 用于暂存各层输出结果
  std::vector<Tensor> temp_buffer_0{tensor_buffer_0_};
  std::vector<Tensor> temp_buffer_1{tensor_buffer_1_};
  std::vector<Tensor> temp_buffer_2{tensor_buffer_2_};

  size_t total_seq_len = 0;
  size_t total_block_num = 0;
  int max_tokens = 0;
  std::vector<int> kv_cache_offset_list(1, 0);

  // prepare inputs
  PrepareKVCache(batch_size, total_seq_len, total_block_num, forward_reqs, kv_cache_offset_list,
                 context_->GetD2HStreams()[rank_], kvcache_offset_event_, false);
  PrepareRotaryEmbeddingPos(batch_size, forward_reqs, context_->GetD2HStreams()[rank_], rotary_embedding_event_);
  PrepareInputIds(batch_size, max_tokens, forward_reqs, context_->GetH2DStreams()[rank_], input_ids_event_);

  // create forward shape tensor
  forward_shape_.shape = {batch_size, max_tokens, kv_cache_offset_list.back()};

  // Forward
  Tensor embedding_weight = base_weight->GetModelWeights("model.embed_tokens.weight");
  std::vector<Tensor>& emb_lookup_output = temp_buffer_0;
  StreamWaitEvent(context_->GetComputeStreams()[rank_], input_ids_event_);
  STATUS_CHECK_RETURN(
      emb_lookup_layer_->Forward({input_ids_, input_offset_uint64_tensor_, embedding_weight}, emb_lookup_output));

  // LlamaDecoder
  for (int layer_idx = 0; layer_idx < num_layer_; ++layer_idx) {
    STATUS_CHECK_RETURN(LlamaDecoder(layer_idx, base_weight, temp_buffer_0, temp_buffer_1, temp_buffer_2,
                                     /*is_context_stage*/ false));
  }

  // final norm
  Tensor final_layernorm_weight = base_weight->GetModelWeights("model.norm.weight");
  std::vector<Tensor>& final_layernorm_input = temp_buffer_0;
  std::vector<Tensor>& final_layernorm_output = temp_buffer_1;
  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({final_layernorm_input[0], final_layernorm_weight}, final_layernorm_output));

  // assemble last token
  std::vector<Tensor>& assemble_last_token_output = temp_buffer_2;
  STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward({final_layernorm_output[0], input_offset_uint64_tensor_},
                                                          assemble_last_token_output));

  // lm_head
  Tensor lm_head_weight = base_weight->GetModelWeights("lm_head.weight");
  std::vector<Tensor>& lm_head_output = temp_buffer_0;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({assemble_last_token_output[0], lm_head_weight}, lm_head_output));

  // Cast to float
  std::vector<Tensor>& logits_float = temp_buffer_1;
  logits_float[0].dtype = TYPE_FP32;
  STATUS_CHECK_RETURN(cast_layer_->Forward(lm_head_output, logits_float));

  CopyToLogistBuffer(batch_size, forward_reqs, logits_float);
#endif
  return Status();
}

template class Llama<float>;
template class Llama<float16>;

}  // namespace ksana_llm
