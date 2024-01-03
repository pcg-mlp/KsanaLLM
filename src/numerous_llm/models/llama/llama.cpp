/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/memory_utils.h"

namespace numerous_llm {

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
}

template <typename T>
Llama<T>::Llama(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context) : context_(context) {
  // 解析 Model Config
  num_layer_ = model_config.num_layer;
  rank_ = rank;
  // TODO: 目前所有层都使用这个 dtype
  weight_data_type_ = model_config.weight_data_type;
  vocab_size_ = model_config.vocab_size;
  float layernorm_eps_ = model_config.layernorm_eps;
  int head_num = model_config.head_num;
  int size_per_head = model_config.size_per_head;
  int hidden_units = size_per_head * head_num;
  int rotary_embedding = model_config.rotary_embedding;
  int num_key_value_heads = model_config.num_key_value_heads;
  int inter_size = model_config.inter_size;
  int max_position_embeddings = model_config.max_position_embeddings;
  float rope_theta = model_config.rope_theta;

  // 1: KV 一块空间
  // 2: 运行时中间空间
  // 3: 矩阵计算需要一块空间

  // input_ids: [max_b, max_s]
  int max_b = model_config.default_batch_size;
  max_seq_len_ = model_config.max_token_num;
  size_t dtype_size = Tensor::GetTypeSize(weight_data_type_);
  CreateTensor(tmp_tensor_0, max_b * max_seq_len_ * hidden_units * dtype_size);
  CreateTensor(tmp_tensor_1, max_b * max_seq_len_ * hidden_units * dtype_size);
  CreateTensor(tmp_tensor_2, max_b * max_seq_len_ * hidden_units * dtype_size);
  CreateTensor(up_matmul_tensor, max_b * max_seq_len_ * inter_size * dtype_size);
  CreateTensor(kv_cache_buffer_, num_layer_ * max_b * max_seq_len_ * 2 * hidden_units * dtype_size);

  // 初始化各层实例
  emb_lookup_layer_ = std::make_shared<EmbLookupLayer>();
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  nccl_all_reduce_sum_layer_ = std::make_shared<NcclAllReduceSumLayer>();
  add_layer_ = std::make_shared<AddLayer>();
  rotary_embedding_layer_ = std::make_shared<RotaryEmbeddingLayer>();
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
  rotary_embedding_layer_->Init({max_position_embeddings, rotary_embedding, rope_theta, size_per_head, head_num,
                                num_key_value_heads, true}, context_, rank_);
  flash_attention_layer_.resize(num_layer_);
  paged_attention_layer_.resize(num_layer_);
  for (int idx = 0; idx < num_layer_; ++idx) {
    flash_attention_layer_[idx] = std::make_shared<FlashAttentionLayer>();
    paged_attention_layer_[idx] = std::make_shared<PagedAttentionLayer>();
    flash_attention_layer_[idx]->Init({idx, max_position_embeddings, head_num, num_key_value_heads, size_per_head},
                                      context_, rank_);
    paged_attention_layer_[idx]->Init({idx, max_position_embeddings, head_num, num_key_value_heads, size_per_head},
                                      context_, rank_);
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
  // 推理前准备三块循环使用的推理时临时空间, 用于暂存各层输出结果
  std::vector<Tensor> output_0{tmp_tensor_0};
  std::vector<Tensor> output_1{tmp_tensor_1};
  std::vector<Tensor> output_2{tmp_tensor_2};
  // 解析外部 CPU 输入,拷贝到 GPU Tensor 中
  size_t total_seq_len = 0;
  size_t total_block_num = 0;
  std::vector<int> kv_cache_offset_list(1, 0);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    total_seq_len += forward_reqs[idx].output_tokens->size();
    total_block_num += forward_reqs[idx].kv_cache_ptrs[rank_].size() / num_layer_;
    kv_cache_offset_list.push_back(total_block_num);
  }

  // create input ids tensor
  // TODO(zezhao): input_ids 复用tmp空间
  Tensor input_ids;
  CreateTensor(input_ids, total_seq_len * sizeof(int));
  input_ids.shape = {total_seq_len};
  input_ids.dtype = TYPE_INT32;
  void* input_ids_ptr = input_ids.GetPtr<void>();
  size_t input_offset = 0;
  std::vector<int> input_offset_list_int32(batch_size + 1, 0);
  std::vector<size_t> input_offset_list_uint64(batch_size + 1, 0ul);
  int max_tokens = 0;
  for (int idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t length = req_input->size();
    // TODO(karlluo): need implement
    // CUDA_CHECK(cudaMemcpyAsync(input_ids_ptr + input_offset, req_input->data(), length * sizeof(int),
    // cudaMemcpyHostToDevice, context->GetH2DStreams()[rank_]));
    CUDA_CHECK(cudaMemcpy(input_ids_ptr + input_offset, req_input->data(), length * sizeof(int),
                          cudaMemcpyHostToDevice));
    input_offset += length;
    input_offset_list_int32[idx + 1] = static_cast<int>(input_offset);
    input_offset_list_uint64[idx + 1] = input_offset;
    max_tokens = std::max(max_tokens, static_cast<int>(length));
  }
  CUDA_CHECK(cudaStreamSynchronize(context_->GetH2DStreams()[rank_]));

  // create input offset tensor int32 and uint64
  Tensor input_offset_int32_tensor, input_offset_uint64_tensor;
  CreateTensor(input_offset_int32_tensor, input_offset_list_int32.size() * sizeof(int));
  CreateTensor(input_offset_uint64_tensor, input_offset_list_uint64.size() * sizeof(size_t));
  input_offset_int32_tensor.shape = {batch_size + 1};
  input_offset_uint64_tensor.shape = {batch_size + 1};
  input_offset_int32_tensor.dtype = TYPE_INT32;
  input_offset_uint64_tensor.dtype = TYPE_UINT64;
  void* input_offset_int32_ptr = input_offset_int32_tensor.GetPtr<void>();
  cudaMemcpy(input_offset_int32_ptr, input_offset_list_int32.data(), (batch_size + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  void* input_offset_uint64_ptr = input_offset_uint64_tensor.GetPtr<void>();
  cudaMemcpy(input_offset_uint64_ptr, input_offset_list_uint64.data(), (batch_size + 1) * sizeof(size_t),
             cudaMemcpyHostToDevice);

  // create kv list tensor
  Tensor kv_list;
  CreateTensor(kv_list, num_layer_ * total_block_num * 2 * sizeof(void*));
  kv_list.shape = {num_layer_, total_block_num * 2};
  kv_list.dtype = TYPE_POINTER;
  std::vector<void*> cpu_kv_list(num_layer_ * total_block_num * 2);
  for (size_t layer_idx = 0; layer_idx < num_layer_; ++layer_idx) {
    int kv_list_index = 0;
    // 处理k
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = forward_reqs[idx].kv_cache_ptrs[rank_].size() / num_layer_;
      for (size_t block_idx = 0; block_idx < block_num; block_idx++){
        cpu_kv_list[layer_idx * total_block_num * 2 + kv_list_index] =
            forward_reqs[idx].kv_cache_ptrs[rank_][layer_idx * block_num + block_idx];
        kv_list_index++;
      }
    }
    // 处理v
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = forward_reqs[idx].kv_cache_ptrs[rank_].size() / num_layer_;
      for (size_t block_idx = 0; block_idx < block_num; block_idx++){
        void * k = forward_reqs[idx].kv_cache_ptrs[rank_][layer_idx * block_num + block_idx];
        size_t block_size = forward_reqs[idx].block_size; //字节数
        cpu_kv_list[layer_idx * total_block_num * 2 + kv_list_index] =
            reinterpret_cast<void *>(reinterpret_cast<char *>(k) + (block_size/2));
        kv_list_index++;
      }
    }
  }
  void* kv_list_ptr = kv_list.GetPtr<void>();
  CUDA_CHECK(cudaMemcpy(kv_list_ptr, cpu_kv_list.data(), cpu_kv_list.size() * sizeof(void*), cudaMemcpyHostToDevice));

  // create rotary embedding pos tensor
  Tensor rotary_embedding_pos;
  CreateTensor(rotary_embedding_pos, sizeof(int64_t) * total_seq_len);
  std::vector<int64_t> cpu_rotary_pos(total_seq_len);
  int cpu_rotary_pos_idx = 0;
  for (size_t idx = 0; idx < batch_size; ++idx) {
    for (size_t pos = 0; pos < forward_reqs[idx].output_tokens->size(); ++pos) {
      cpu_rotary_pos[cpu_rotary_pos_idx++] = pos;
    }
  }
  void* rotary_embedding_pos_ptr = rotary_embedding_pos.GetPtr<void>();
  CUDA_CHECK(cudaMemcpy(rotary_embedding_pos_ptr, cpu_rotary_pos.data(), sizeof(int64_t) * total_seq_len,
                        cudaMemcpyHostToDevice));

  // create forward shape tensor
  Tensor forward_shape;
  CreateTensor(forward_shape, sizeof(int));
  forward_shape.shape = {batch_size, max_tokens};

  // Forward
  // embedding
  Tensor embedding_weight = base_weight->GetModelWeights("gather_embedding");
  std::vector<Tensor>& emb_lookup_output = output_0;
  STATUS_CHECK_RETURN(emb_lookup_layer_->Forward({input_ids, input_offset_uint64_tensor, embedding_weight},
                                                 emb_lookup_output));
  emb_lookup_output[0].SaveToFile(saved_dir + "emb_lookup_output.npy");
  NLLM_LOG_INFO << "embeddig";

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

    input_layernorm_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".input_layernorm.npy");
    NLLM_LOG_INFO << "input layernorm";

    // Attn proj MatMul
    Tensor attn_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".attention.query_key_value");
    std::vector<Tensor>& attn_proj_output = output_2;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({input_layernorm_output[0], attn_proj_weight}, attn_proj_output));

    attn_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.proj.npy");
    NLLM_LOG_INFO << "attn proj";

    // rotary embedding 原地修改, 因此输出和输入指向同一块空间
    std::vector<Tensor>& rotary_embedding_output = output_2;
    STATUS_CHECK_RETURN(rotary_embedding_layer_->Forward({attn_proj_output[0], rotary_embedding_pos,
                                                         forward_shape}, rotary_embedding_output));
    rotary_embedding_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.rotary_embedding.npy");
    NLLM_LOG_INFO << "rotary embedding";

    // MMHA Flash Attention
    std::vector<Tensor>& flash_attention_output = output_1;
    STATUS_CHECK_RETURN(flash_attention_layer_[layer_num]->Forward({rotary_embedding_output[0], input_offset_uint64_tensor, kv_list,
                                                                   kv_cache_buffer_}, flash_attention_output));
    flash_attention_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.MMHA.npy");
    NLLM_LOG_INFO << "MMHA Flash Attention";

    // Attn o_proj MatMul
    Tensor attn_o_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".attention.dense");
    std::vector<Tensor>& attn_o_proj_output = output_2;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({flash_attention_output[0], attn_o_proj_weight}, attn_o_proj_output));
    attn_o_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.o_proj.npy");

    // Attn NcclAllReduceSum
    std::vector<Tensor>& attn_all_reduce_sum_output = output_1;
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(attn_o_proj_output, attn_all_reduce_sum_output));
    attn_o_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".attn_all_reduce_sum.npy");

    // Attn Add
    std::vector<Tensor>& attn_add_output = output_2;
    STATUS_CHECK_RETURN(
        add_layer_->Forward({input_layernorm_input[0], attn_all_reduce_sum_output[0]}, attn_add_output));
    attn_add_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.add.npy");

    // post_attention_layernorm
    Tensor post_layernorm_weight =
        base_weight->GetModelWeights(std::to_string(layer_num) + ".post_attention_layernorm");
    std::vector<Tensor>& post_layernorm_output = output_1;
    STATUS_CHECK_RETURN(layernorm_layer_->Forward({attn_add_output[0], post_layernorm_weight}, post_layernorm_output));
    post_layernorm_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".post_attention_layernorm.npy");

    // Mlp gate_proj MatMul
    Tensor gate_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.gate_proj");
    std::vector<Tensor>& gate_matmul_output = output_0;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output[0], gate_proj_weight}, gate_matmul_output));
    gate_matmul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.gate_proj.npy");

    // Mlp up_proj MatMul 由于 gate_proj 与 up_proj 为并行关系,因此此处使用额外空间存储 matmul 结果
    Tensor up_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.up_proj");
    std::vector<Tensor> up_matmul_output = {up_matmul_tensor};
    STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output[0], up_proj_weight}, up_matmul_output));
    up_matmul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.up_proj.npy");

    std::vector<Tensor>& silu_mul_output = output_1;
    STATUS_CHECK_RETURN(silu_mul_layer_->Forward({gate_matmul_output[0], up_matmul_output[0]}, silu_mul_output));
    silu_mul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.silu.npy");

    // Mlp down_proj MatMul
    Tensor down_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.down_proj");
    std::vector<Tensor>& down_proj_output = output_0;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({silu_mul_output[0], down_proj_weight}, down_proj_output));
    down_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.down_proj.npy");

    // Mlp NcclAllReduceSum
    std::vector<Tensor>& mlp_all_reduce_sum_output = output_1;
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward({down_proj_output[0]}, mlp_all_reduce_sum_output));
    mlp_all_reduce_sum_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.nccl_all_reducesum.npy");

    // Mlp Add
    std::vector<Tensor>& mlp_add_output = output_0;
    STATUS_CHECK_RETURN(add_layer_->Forward({mlp_all_reduce_sum_output[0], attn_add_output[0]}, mlp_add_output));
    mlp_add_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.add.npy");
  }
  // final norm
  Tensor final_layernorm_weight = base_weight->GetModelWeights("norm");
  std::vector<Tensor>& final_layernorm_input = output_0;
  std::vector<Tensor>& final_layernorm_output = output_1;
  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({final_layernorm_input[0], final_layernorm_weight}, final_layernorm_output));
  final_layernorm_output[0].SaveToFile(saved_dir + "final_norm.npy");

  // assemble last token
  std::vector<Tensor>& assemble_last_token_output = output_2;
  STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward({final_layernorm_output[0], input_offset_uint64_tensor},
                      assemble_last_token_output));
  assemble_last_token_output[0].SaveToFile(saved_dir + "assemble_last_token.npy");

  // lm_head
  Tensor lm_head_weight = base_weight->GetModelWeights("lm_head");
  lm_head_weight.SaveToFile(saved_dir + "lm_head.weight.npy");
  std::vector<Tensor>& lm_head_output = output_0;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({assemble_last_token_output[0], lm_head_weight}, lm_head_output));
  lm_head_output[0].SaveToFile(saved_dir + "lm_head.npy");

  // Cast to float
  std::vector<Tensor>& logits_float = output_1;
  logits_float[0].dtype = TYPE_FP32;
  STATUS_CHECK_RETURN(cast_layer_->Forward(lm_head_output, logits_float));
  logits_float[0].SaveToFile(saved_dir + "logits_float.npy");

  // Copy to logits buf
  float* logits_ptr = logits_float[0].GetPtr<float>();
  for (int idx = 0; idx < batch_size; ++idx) {
    ForwardRequest& req = forward_reqs[idx];
    req.logits_buf[rank_] = logits_ptr;
    req.logits_offset = idx;
  }

  DestroyTensor(input_ids);
  DestroyTensor(input_offset_uint64_tensor);
  DestroyTensor(input_offset_int32_tensor);
  DestroyTensor(kv_list);
  DestroyTensor(rotary_embedding_pos);
  DestroyTensor(forward_shape);
  return Status();
}

template <typename T>
Status Llama<T>::Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest>& forward_reqs) {
  NLLM_LOG_INFO << "llama decode stage inference";

  saved_dir  = "/model/llama-ft/7B/nllm_decode/";
  
  size_t batch_size = forward_reqs.size();
  // 推理前准备三块循环使用的推理时临时空间, 用于暂存各层输出结果
  std::vector<Tensor> output_0{tmp_tensor_0};
  std::vector<Tensor> output_1{tmp_tensor_1};
  std::vector<Tensor> output_2{tmp_tensor_2};
  // 解析外部 CPU 输入,拷贝到 GPU Tensor 中
  size_t total_seq_len = 0;
  size_t total_block_num = 0;
  std::vector<int> kv_cache_offset_list(1, 0);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    total_seq_len += forward_reqs[idx].output_tokens->size();
    total_block_num += forward_reqs[idx].kv_cache_ptrs[rank_].size() / num_layer_;
    kv_cache_offset_list.push_back(total_block_num);
  }

  // create input ids tensor
  // TODO(zezhao): input_ids 复用tmp空间
  Tensor input_ids;
  CreateTensor(input_ids, total_seq_len * sizeof(int));
  input_ids.shape = {total_seq_len};
  input_ids.dtype = TYPE_INT32;
  void* input_ids_ptr = input_ids.GetPtr<void>();
  size_t input_offset = 0;
  std::vector<int> input_offset_list_int32(batch_size + 1, 0);
  std::vector<size_t> input_offset_list_uint64(batch_size + 1, 0ul);
  int max_tokens = 0;
  for (int idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t length = req_input->size();
    // TODO(karlluo): need implement
    // CUDA_CHECK(cudaMemcpyAsync(input_ids_ptr + input_offset, req_input->data(), length * sizeof(int),
    // cudaMemcpyHostToDevice, context->GetH2DStreams()[rank_]));
    CUDA_CHECK(cudaMemcpy(input_ids_ptr + input_offset, req_input->data(), length * sizeof(int),
                          cudaMemcpyHostToDevice));
    input_offset += length;
    input_offset_list_int32[idx + 1] = static_cast<int>(input_offset);
    input_offset_list_uint64[idx + 1] = input_offset;
    max_tokens = std::max(max_tokens, static_cast<int>(length));
  }
  CUDA_CHECK(cudaStreamSynchronize(context_->GetH2DStreams()[rank_]));

  // create input offset tensor int32 and uint64
  Tensor input_offset_int32_tensor, input_offset_uint64_tensor;
  CreateTensor(input_offset_int32_tensor, input_offset_list_int32.size() * sizeof(int));
  CreateTensor(input_offset_uint64_tensor, input_offset_list_uint64.size() * sizeof(size_t));
  input_offset_int32_tensor.shape = {batch_size + 1};
  input_offset_uint64_tensor.shape = {batch_size + 1};
  input_offset_int32_tensor.dtype = TYPE_INT32;
  input_offset_uint64_tensor.dtype = TYPE_UINT64;
  void* input_offset_int32_ptr = input_offset_int32_tensor.GetPtr<void>();
  cudaMemcpy(input_offset_int32_ptr, input_offset_list_int32.data(), (batch_size + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  void* input_offset_uint64_ptr = input_offset_uint64_tensor.GetPtr<void>();
  cudaMemcpy(input_offset_uint64_ptr, input_offset_list_uint64.data(), (batch_size + 1) * sizeof(size_t),
             cudaMemcpyHostToDevice);

  // create kv list tensor
  Tensor kv_list;
  CreateTensor(kv_list, num_layer_ * total_block_num * 2 * sizeof(void*));
  kv_list.shape = {num_layer_, total_block_num * 2};
  kv_list.dtype = TYPE_POINTER;
  std::vector<void*> cpu_kv_list(num_layer_ * total_block_num * 2);
  for (size_t layer_idx = 0; layer_idx < num_layer_; ++layer_idx) {
    int kv_list_index = 0;
    // 处理k
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = forward_reqs[idx].kv_cache_ptrs[rank_].size() / num_layer_;
      for (size_t block_idx = 0; block_idx < block_num; block_idx++){
        cpu_kv_list[layer_idx * total_block_num * 2 + kv_list_index] =
            forward_reqs[idx].kv_cache_ptrs[rank_][layer_idx * block_num + block_idx];
        kv_list_index++;
      }
    }
    // 处理v
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = forward_reqs[idx].kv_cache_ptrs[rank_].size() / num_layer_;
      for (size_t block_idx = 0; block_idx < block_num; block_idx++){
        void * k = forward_reqs[idx].kv_cache_ptrs[rank_][layer_idx * block_num + block_idx];
        size_t block_size = forward_reqs[idx].block_size; //字节数
        cpu_kv_list[layer_idx * total_block_num * 2 + kv_list_index] =
            reinterpret_cast<void *>(reinterpret_cast<char *>(k) + (block_size/2));
        kv_list_index++;
      }
    }
  }
  void* kv_list_ptr = kv_list.GetPtr<void>();
  CUDA_CHECK(cudaMemcpy(kv_list_ptr, cpu_kv_list.data(), cpu_kv_list.size() * sizeof(void*), cudaMemcpyHostToDevice));

  // create rotary embedding pos tensor
  Tensor rotary_embedding_pos;
  CreateTensor(rotary_embedding_pos, sizeof(int64_t) * total_seq_len);
  std::vector<int64_t> cpu_rotary_pos(total_seq_len);
  int cpu_rotary_pos_idx = 0;
  for (size_t idx = 0; idx < batch_size; ++idx) {
    for (size_t pos = 0; pos < forward_reqs[idx].output_tokens->size(); ++pos) {
      cpu_rotary_pos[cpu_rotary_pos_idx++] = pos;
    }
  }
  void* rotary_embedding_pos_ptr = rotary_embedding_pos.GetPtr<void>();
  CUDA_CHECK(cudaMemcpy(rotary_embedding_pos_ptr, cpu_rotary_pos.data(), sizeof(int64_t) * total_seq_len,
                        cudaMemcpyHostToDevice));

  // create forward shape tensor
  Tensor forward_shape;
  CreateTensor(forward_shape, sizeof(int));
  forward_shape.shape = {batch_size, max_tokens};

  // Forward
  // embedding
  Tensor embedding_weight = base_weight->GetModelWeights("gather_embedding");
  std::vector<Tensor>& emb_lookup_output = output_0;
  STATUS_CHECK_RETURN(emb_lookup_layer_->Forward({input_ids, input_offset_uint64_tensor, embedding_weight},
                                                 emb_lookup_output));
  emb_lookup_output[0].SaveToFile(saved_dir + "emb_lookup_output.npy");
  NLLM_LOG_INFO << "embeddig";

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

    input_layernorm_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".input_layernorm.npy");
    NLLM_LOG_INFO << "input layernorm";

    // Attn proj MatMul
    Tensor attn_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".attention.query_key_value");
    std::vector<Tensor>& attn_proj_output = output_2;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({input_layernorm_output[0], attn_proj_weight}, attn_proj_output));

    attn_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.proj.npy");
    NLLM_LOG_INFO << "attn proj";

    // rotary embedding 原地修改, 因此输出和输入指向同一块空间
    std::vector<Tensor>& rotary_embedding_output = output_2;
    STATUS_CHECK_RETURN(rotary_embedding_layer_->Forward({attn_proj_output[0], rotary_embedding_pos,
                                                         forward_shape}, rotary_embedding_output));
    rotary_embedding_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.rotary_embedding.npy");
    NLLM_LOG_INFO << "rotary embedding";

    // MMHA Paged Attention
    std::vector<Tensor>& paged_attention_output = output_1;
    paged_attention_output.push_back(kv_list);
    paged_attention_output.push_back(up_matmul_tensor);
    // TODO(zezhao): workspace 需与 catheywang 确认传入大小,暂时使用此处用不到的空间跑通流程
    Tensor kv_cache_offset_tensor;
    CreateTensor(kv_cache_offset_tensor, kv_cache_offset_list.size() * sizeof(int));
    kv_cache_offset_tensor.shape = {kv_cache_offset_list.size()};
    kv_cache_offset_tensor.dtype = TYPE_INT32;
    void* kv_cache_offset_ptr = kv_cache_offset_tensor.GetPtr<void>();
    CUDA_CHECK(cudaMemcpy(kv_cache_offset_ptr, kv_cache_offset_list.data(),
                          kv_cache_offset_list.size() * sizeof(int), cudaMemcpyHostToDevice));
    STATUS_CHECK_RETURN(paged_attention_layer_[layer_num]->Forward({rotary_embedding_output[0], input_offset_uint64_tensor,
                                                                    kv_cache_offset_tensor, forward_shape},
                                                                   paged_attention_output));
    DestroyTensor(kv_cache_offset_tensor);
    paged_attention_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.MMHA.npy");
    NLLM_LOG_INFO << "MMHA Paged Attention";

    // Attn o_proj MatMul
    Tensor attn_o_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".attention.dense");
    std::vector<Tensor>& attn_o_proj_output = output_2;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({paged_attention_output[0], attn_o_proj_weight}, attn_o_proj_output));
    attn_o_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.o_proj.npy");

    // Attn NcclAllReduceSum
    std::vector<Tensor>& attn_all_reduce_sum_output = output_1;
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(attn_o_proj_output, attn_all_reduce_sum_output));

    // Attn Add
    std::vector<Tensor>& attn_add_output = output_2;
    STATUS_CHECK_RETURN(
        add_layer_->Forward({input_layernorm_input[0], attn_all_reduce_sum_output[0]}, attn_add_output));
    attn_add_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".self_attn.add.npy");

    // post_attention_layernorm
    Tensor post_layernorm_weight =
        base_weight->GetModelWeights(std::to_string(layer_num) + ".post_attention_layernorm");
    std::vector<Tensor>& post_layernorm_output = output_1;
    STATUS_CHECK_RETURN(layernorm_layer_->Forward({attn_add_output[0], post_layernorm_weight}, post_layernorm_output));
    post_layernorm_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".post_attention_layernorm.npy");

    // Mlp gate_proj MatMul
    Tensor gate_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.gate_proj");
    std::vector<Tensor>& gate_matmul_output = output_0;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output[0], gate_proj_weight}, gate_matmul_output));
    gate_matmul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.gate_proj.npy");

    // Mlp up_proj MatMul 由于 gate_proj 与 up_proj 为并行关系,因此此处使用额外空间存储 matmul 结果
    Tensor up_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.up_proj");
    std::vector<Tensor> up_matmul_output = {up_matmul_tensor};
    STATUS_CHECK_RETURN(matmul_layer_->Forward({post_layernorm_output[0], up_proj_weight}, up_matmul_output));
    up_matmul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.up_proj.npy");

    std::vector<Tensor>& silu_mul_output = output_1;
    STATUS_CHECK_RETURN(silu_mul_layer_->Forward({gate_matmul_output[0], up_matmul_output[0]}, silu_mul_output));
    silu_mul_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.silu.npy");

    // Mlp down_proj MatMul
    Tensor down_proj_weight = base_weight->GetModelWeights(std::to_string(layer_num) + ".mlp.down_proj");
    std::vector<Tensor>& down_proj_output = output_0;
    STATUS_CHECK_RETURN(matmul_layer_->Forward({silu_mul_output[0], down_proj_weight}, down_proj_output));
    down_proj_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.down_proj.npy");

    // Mlp NcclAllReduceSum
    std::vector<Tensor>& mlp_all_reduce_sum_output = output_1;
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward({down_proj_output[0]}, mlp_all_reduce_sum_output));
    mlp_all_reduce_sum_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.nccl_all_reducesum.npy");

    // Mlp Add
    std::vector<Tensor>& mlp_add_output = output_0;
    STATUS_CHECK_RETURN(add_layer_->Forward({mlp_all_reduce_sum_output[0], attn_add_output[0]}, mlp_add_output));
    mlp_add_output[0].SaveToFile(saved_dir + std::to_string(layer_num) + ".mlp.add.npy");
  }
  // final norm
  Tensor final_layernorm_weight = base_weight->GetModelWeights("norm");
  std::vector<Tensor>& final_layernorm_input = output_0;
  std::vector<Tensor>& final_layernorm_output = output_1;
  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({final_layernorm_input[0], final_layernorm_weight}, final_layernorm_output));
  final_layernorm_output[0].SaveToFile(saved_dir + "final_norm.npy");

  // assemble last token
  std::vector<Tensor>& assemble_last_token_output = output_2;
  STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward({final_layernorm_output[0], input_offset_uint64_tensor},
                      assemble_last_token_output));
  assemble_last_token_output[0].SaveToFile(saved_dir + "assemble_last_token.npy");

  // lm_head
  Tensor lm_head_weight = base_weight->GetModelWeights("lm_head");
  lm_head_weight.SaveToFile(saved_dir + "lm_head.weight.npy");
  std::vector<Tensor>& lm_head_output = output_0;
  STATUS_CHECK_RETURN(matmul_layer_->Forward({assemble_last_token_output[0], lm_head_weight}, lm_head_output));
  lm_head_output[0].SaveToFile(saved_dir + "lm_head.npy");

  // Copy to logits buf
  float* output_logits_buf = forward_reqs[0].logits_buf[rank_];
  size_t logits_offset = 0;
  // TODO(karlluo): need implement
  // CUDA_CHECK(cudaMemcpyAsync(output_logits_buf, lm_head_output[0].GetPtr<void>(), lm_head_output[0].GetTotalBytes(),
  //            cudaMemcpyDeviceToDevice, context_->GetD2DStreams()[rank_]));
  CUDA_CHECK(cudaStreamSynchronize(context_->GetD2DStreams()[rank_]));
  for (int idx = 0; idx < batch_size; ++idx) {
    ForwardRequest req = forward_reqs[idx];
    logits_offset += req.output_tokens->size();
    req.logits_offset = logits_offset - 1;
  }

  DestroyTensor(input_ids);
  DestroyTensor(input_offset_int32_tensor);
  DestroyTensor(input_offset_uint64_tensor);
  DestroyTensor(kv_list);
  DestroyTensor(forward_shape);
  DestroyTensor(rotary_embedding_pos);
  return Status();
}

template class Llama<float>;
template class Llama<half>;

}  // namespace numerous_llm
