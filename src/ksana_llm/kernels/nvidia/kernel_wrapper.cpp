/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

#include <fstream>
#include <iostream>

#ifdef ENABLE_FLASH_ATTN_2
#  include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#else
#  include "flash_api.h"
#endif

#include "csrc/kernels/nvidia/activation/activation.h"
#include "csrc/kernels/nvidia/add/add.h"
#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "csrc/kernels/nvidia/assemble_last_token/assemble_last_token.h"
#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/paged_attention.h"
#include "csrc/kernels/nvidia/permute/permute.h"

#include "ksana_llm/kernels/cast.h"

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"

namespace ksana_llm {

Status LookupEmbedding(const void* ids, const void* offset, const void* emb, const void* pos, void* output,
                       int vocab_size, int hidden_size, int bs, int step, int vocab_id, Stream& stream,
                       void* workspace_ptr) {
  llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<half>(
      reinterpret_cast<half*>(output), reinterpret_cast<const half*>(emb), reinterpret_cast<const half*>(pos), {},
      reinterpret_cast<const int32_t*>(ids), step, reinterpret_cast<const size_t*>(offset), bs, hidden_size, vocab_size,
      vocab_id, stream.Get());

  return Status();
}

void InvokeLayerNorm(const void* input, const void* weight, const float layernorm_eps, const int m, const int n,
                     void* output, cudaStream_t stream) {
  half* beta = nullptr;
  llm_kernels::nvidia::InvokeLayerNorm<half>(reinterpret_cast<half*>(output), reinterpret_cast<const half*>(input),
                                             reinterpret_cast<const half*>(weight), beta, layernorm_eps, m, n, stream);
}

void InvokeMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                  const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream) {
  CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(cublas_handle, cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                                   b_ptr, n, CUDA_R_16F, a_ptr, k, CUDA_R_16F, c_ptr, n, CUDA_R_16F,
                                                   CUDA_R_32F, stream));
}

void InvokeAddBiasResidual(const void* input_a, const void* input_b, const void* bias, const int m, const int n,
                           void* output, cudaStream_t stream) {
  llm_kernels::nvidia::InvokeAddBiasResidual<half>(
      reinterpret_cast<half*>(output), reinterpret_cast<const half*>(input_a), reinterpret_cast<const half*>(input_b),
      nullptr, reinterpret_cast<const half*>(bias), nullptr, nullptr, m, n, stream);
}

void InvokeSiluActivation(const void* input, const void* gated_weights, const int m, const int n, void* output,
                          cudaStream_t stream) {
  const int* ia3_tasks = nullptr;
  const half* bias = nullptr;
  const half* ia3_weights = nullptr;
  const half* gated_bias = nullptr;
  const int int8_mode = 0;
  const int* padding_offset = nullptr;
  const int seq_len = 0;
  const float* activation_in = nullptr;
  const float* activation_out = nullptr;
  CUDA_CHECK(cudaMemcpyAsync(output, input, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream));
  llm_kernels::nvidia::InvokeGenericActivation<llm_kernels::nvidia::SiluActivation, half, half>(
      reinterpret_cast<half*>(output), bias, reinterpret_cast<const half*>(gated_weights), gated_bias, ia3_tasks,
      ia3_weights, m, n, int8_mode, activation_in, activation_out, padding_offset, seq_len, stream);
}

void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* out, void* seqlen,
                 llm_kernels::nvidia::RotaryEmbeddingCuda<half>& rotary_embedding_cuda, int total_tokens,
                 int max_tokens, int batch, int num_heads, int head_size, int stride_size, int tensor_para_size,
                 bool is_causal, int rank, int block_size, void** k_list, void** v_list, void* block_offset,
                 const std::optional<void*>& alibi_slopes, cudaStream_t stream) {
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat16);
  torch::Tensor qkv_tensor = torch::from_blob(qkv_ptr, {total_tokens, num_heads * head_size * 3}, options);
  auto tt = qkv_tensor.split(qkv_tensor.size(-1) / 3, -1);

  c10::optional<at::Tensor> out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);
  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);
  torch::Tensor seqlen_tensor = torch::from_blob(seqlen, {batch + 1}, int_options);

  // rotary embedding
  torch::Tensor q_tensor = tt[0];
  torch::Tensor k_tensor = tt[1];
  torch::Tensor v_tensor = tt[2];

  if (!alibi_slopes.has_value()) {
    rotary_embedding_cuda.SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                   reinterpret_cast<half*>(q_tensor.data_ptr()),
                                   reinterpret_cast<half*>(k_tensor.data_ptr()), total_tokens, stream);
    rotary_embedding_cuda.Forward();
  }

  llm_kernels::nvidia::CacheCopy<half>(reinterpret_cast<half*>(k_tensor.data_ptr()),
                                       reinterpret_cast<half*>(v_tensor.data_ptr()), k_list, v_list,
                                       reinterpret_cast<size_t*>(seqlen), reinterpret_cast<int*>(block_offset),
                                       block_size, batch, total_tokens, num_heads, head_size, stride_size, stream);

// flash attention 2 or flash attention 1
#ifdef ENABLE_FLASH_ATTN_2
  at::Tensor q_tmp_tensor = torch::reshape(q_tensor, {total_tokens, num_heads, head_size});
  c10::optional<at::Tensor> seqused_k = c10::nullopt;
  auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
  c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
  if (alibi_slopes.has_value()) {
    alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
  }
  mha_varlen_fwd(q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_heads, head_size}),
                 torch::reshape(tt[2], {total_tokens, num_heads, head_size}), out_tensor,
                 seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), seqused_k, alibi_slopes_tensor,
                 max_tokens, max_tokens, 0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, false, c10::nullopt);
#else
  flash_attn::mha_varlen_fwd(torch::reshape(q_tensor, {total_tokens, num_heads, head_size}),
                             torch::reshape(k_tensor, {total_tokens, num_heads, head_size}),
                             torch::reshape(tt[2], {total_tokens, num_heads, head_size}), out_tensor,
                             seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), max_tokens, max_tokens,
                             0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, false, c10::nullopt);
#endif
}

template <typename T>
void run_paged_attention(void* out,                // [num_seqs, num_heads, head_size]
                         void* query,              // [num_seqs, num_heads, head_size]
                         void** key_cache_ptrs,    // num_seqs,[seq_blocks]
                         void** value_cache_ptrs,  // num_seqs,[seq_blocks]
                         void* context_lens_ptr,   // [num_seqs]
                         int max_context_len, cudaStream_t stream,
                         void* cache_offsets_ptr,  // num_seqs
                         int num_seqs, int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size,
                         int batch, void* rotary_embedding_pos, int total_tokens,
                         llm_kernels::nvidia::RotaryEmbeddingCuda<half>& rotary_embedding_cuda, void* workspace,
                         size_t work_size, int rank, const std::optional<void*>& alibi_slopes, void* qkv_workspace) {
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat16);
  torch::Tensor qkv_tensor = torch::from_blob(query, {total_tokens, num_heads * head_size * 3}, options);
  auto tt = qkv_tensor.split(qkv_tensor.size(-1) / 3, -1);

  torch::Tensor q_tensor = tt[0];
  torch::Tensor k_tensor = tt[1];
  torch::Tensor v_tensor = tt[2];
  void* q_tensor_ptr = q_tensor.data_ptr();
  void* k_tensor_ptr = k_tensor.data_ptr();
  void* v_tensor_ptr = v_tensor.data_ptr();

  if (!alibi_slopes.has_value()) {
    // When the alibi_slopes parameter is empty, execute the rotary embedding.
    rotary_embedding_cuda.SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                   reinterpret_cast<half*>(q_tensor_ptr), reinterpret_cast<half*>(k_tensor_ptr),
                                   total_tokens, stream);
    rotary_embedding_cuda.Forward();
  }

  llm_kernels::nvidia::CachePosCopy<half>(
      reinterpret_cast<half*>(k_tensor_ptr), reinterpret_cast<half*>(v_tensor_ptr), key_cache_ptrs, value_cache_ptrs,
      rotary_embedding_pos, reinterpret_cast<size_t*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr),
      block_size, batch, total_tokens, num_heads, head_size, stride_size, stream);

  llm_kernels::nvidia::PagedAttentionCuda<uint16_t> op;
  op.SetConfig(num_kv_heads, num_heads, head_size, block_size, stride_size);

  op.SetInput(reinterpret_cast<uint16_t*>(out), reinterpret_cast<const uint16_t*>(q_tensor_ptr),
              reinterpret_cast<uint16_t**>(key_cache_ptrs), reinterpret_cast<uint16_t**>(value_cache_ptrs),
              reinterpret_cast<const int*>(cache_offsets_ptr), reinterpret_cast<const int*>(context_lens_ptr),
              max_context_len, num_seqs, stream, workspace, work_size, alibi_slopes_ptr);
  op.Forward();
}

template void run_paged_attention<half>(void* out,                // [num_seqs, num_heads, head_size]
                                        void* query,              // [num_seqs, num_heads, head_size]
                                        void** key_cache_ptrs,    // num_seqs,[seq_blocks]
                                        void** value_cache_ptrs,  // num_seqs,[seq_blocks]
                                        void* context_lens_ptr,   // [num_seqs]
                                        int max_context_len, cudaStream_t stream,
                                        void* cache_offsets_ptr,  // num_seqs
                                        int num_seqs, int num_heads, int head_size, int num_kv_heads, int stride_size,
                                        int block_size, int batch, void* rotary_embedding_pos, int total_tokens,
                                        llm_kernels::nvidia::RotaryEmbeddingCuda<half>& rotary_embedding_cuda,
                                        void* workspace, size_t work_size, int rank,
                                        const std::optional<void*>& alibi_slopes, void* qkv_workspace);

void AssembleLastToken(const void* input, const void* offset, const int batch_size, const int hidden_units_num,
                       void* output, cudaStream_t& stream) {
  llm_kernels::nvidia::AssembleLastToken<half>(reinterpret_cast<const half*>(input),
                                               reinterpret_cast<const size_t*>(offset), batch_size, hidden_units_num,
                                               reinterpret_cast<half*>(output), stream);
}

void HalfToFloat(const void* input, const int data_size, void* output, cudaStream_t& stream) {
  llm_kernels::nvidia::HalfToFloat(reinterpret_cast<const half*>(input), data_size, reinterpret_cast<float*>(output),
                                   stream);
}

void BFloat16ToFloat16(void* data_ptr, const int data_size, cudaStream_t& stream) {
#ifdef ENABLE_CUDA
  llm_kernels::nvidia::BFP16ToFP16(data_ptr, data_size, stream);
#endif
}

void CustomAllReduceInit(void** ptr, void* input, void** metas, void* rank_data, void** data_handles,
                         void** input_handles, int data_size, size_t rank_data_sz, int tp_size, int rank,
                         cudaStream_t& stream) {
  std::vector<int64_t> offsets(tp_size, 0);
  *ptr = new llm_kernels::nvidia::CustomAllreduce(metas, rank_data, rank_data_sz, data_handles, offsets, rank);
  llm_kernels::nvidia::CustomAllreduce* reduce_op = static_cast<llm_kernels::nvidia::CustomAllreduce*>(*ptr);
  // hack buffer registration
  if (input != input_handles[rank]) {
    NLLM_LOG_ERROR << "input != input_handles[rank]";
  }
  std::vector<std::string> handles;
  handles.reserve(tp_size);
  for (int i = 0; i < tp_size; i++) {
    char* begin = (char*)&(input_handles[i]);
    char* end = (char*)&(input_handles[i + 1]);
    handles.emplace_back(begin, end);
  }
  reduce_op->RegisterBuffer(handles, offsets, input, stream);
}

void CustomAllReduceRun(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream) {
  llm_kernels::nvidia::CustomAllreduce* reduce_op = static_cast<llm_kernels::nvidia::CustomAllreduce*>(ptr);
  reduce_op->AllReduce<half>(stream, static_cast<half*>(input), static_cast<half*>(result), data_size);
}

void InvokePermute(void* input, void* output, std::vector<size_t> input_shape, std::vector<size_t> permutation,
                   cudaStream_t& stream) {
  NLLM_CHECK_WITH_INFO(input_shape.size() <= 4ul,
                       fmt::format("input shape dims number {} > 4 is not supported", input_shape.size()));
  if (input_shape.empty()) {
    return;
  }

  // Extend to num_dims = 4
  input_shape.resize(4, 1);
  for (size_t i = permutation.size(); i < 4; ++i) {
    permutation.push_back(i);
  }
  llm_kernels::nvidia::InvokePermute<4ul, sizeof(half)>(input, output, input_shape, permutation, stream);
}

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  if (tensor.dtype == DataType::TYPE_BF16 && target_dtype == DataType::TYPE_FP16) {
    BFloat16ToFloat16(tensor.GetPtr<void>(), tensor.GetElementNumber(), stream.Get());
  } else {
    throw std::runtime_error(
        fmt::format("CastInplace from type {} to {} is not yet implement", tensor.dtype, target_dtype));
  }
  tensor.dtype = DataType::TYPE_FP16;
  return Status();
}

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* workspace_ptr) {
  InvokePermute(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape, permutation,
                stream.Get());
  return Status();
}

// c = Mul(a, b)
void Mul(float* a, float* b, float* c, int n, int device_rank) {
  auto options = torch::TensorOptions().device(torch::kCUDA, device_rank).dtype(torch::kFloat32);
  auto a_tensor = torch::from_blob(a, {n}, options);
  auto b_tensor = torch::from_blob(b, {n}, options);
  auto c_tensor = torch::from_blob(c, {n}, options);
  mul_out(c_tensor, a_tensor, b_tensor);
}

}  // namespace ksana_llm
