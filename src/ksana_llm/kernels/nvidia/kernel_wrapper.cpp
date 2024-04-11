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

template <typename T>
void LookupEmbedding(const void* ids, const void* offset, const void* emb, const void* pos, void* output,
                     int vocab_size, int hidden_size, int bs, int step, int vocab_id, cudaStream_t stream,
                     void* workspace_ptr) {
  llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), reinterpret_cast<const T*>(pos), {},
      reinterpret_cast<const int32_t*>(ids), step, reinterpret_cast<const size_t*>(offset), bs, hidden_size, vocab_size,
      vocab_id, stream);
}
#define LOOKUP_EMBEDDING(T)                                                                                       \
  template void LookupEmbedding<T>(const void* ids, const void* offset, const void* emb, const void* pos,         \
                                   void* output, int vocab_size, int hidden_size, int bs, int step, int vocab_id, \
                                   cudaStream_t stream, void* workspace_ptr)
LOOKUP_EMBEDDING(float);
LOOKUP_EMBEDDING(half);
#ifdef ENABLE_BFLOAT16
LOOKUP_EMBEDDING(__nv_bfloat16);
#endif
#undef LOOKUP_EMBEDDING

template <typename T>
void InvokeLayerNorm(const void* input, const void* weight, const float layernorm_eps, const int m, const int n,
                     void* output, cudaStream_t stream) {
  T* beta = nullptr;
  llm_kernels::nvidia::InvokeLayerNorm<T>(reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input),
                                          reinterpret_cast<const T*>(weight), beta, layernorm_eps, m, n, stream);
}
#define INVOKE_LAYER_NORM(T)                                                                                      \
  template void InvokeLayerNorm<T>(const void* input, const void* weight, const float layernorm_eps, const int m, \
                                   const int n, void* output, cudaStream_t stream)
INVOKE_LAYER_NORM(float);
INVOKE_LAYER_NORM(half);
#ifdef ENABLE_BFLOAT16
INVOKE_LAYER_NORM(__nv_bfloat16);
#endif
#undef INVOKE_LAYER_NORM

#define INVOKE_MATMUL(T, CUDA_TYPE)                                                                                    \
  template <>                                                                                                          \
  void InvokeMatMul<T>(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,            \
                       const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream) {                      \
    CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(cublas_handle, cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m,   \
                                                     k, b_ptr, n, CUDA_TYPE, a_ptr, k, CUDA_TYPE, c_ptr, n, CUDA_TYPE, \
                                                     CUDA_R_32F, stream));                                             \
  }
INVOKE_MATMUL(float, CUDA_R_32F);
INVOKE_MATMUL(half, CUDA_R_16F);
#ifdef ENABLE_BFLOAT16
INVOKE_MATMUL(__nv_bfloat16, CUDA_R_16BF);
#endif
#undef INVOKE_MATMUL

template <typename T>
void InvokeAddBiasResidual(const void* input_a, const void* input_b, const void* bias, const int m, const int n,
                           void* output, cudaStream_t stream) {
  llm_kernels::nvidia::InvokeAddBiasResidual<T>(reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input_a),
                                                reinterpret_cast<const T*>(input_b), nullptr,
                                                reinterpret_cast<const T*>(bias), nullptr, nullptr, m, n, stream);
}

#define INVOKE_ADD_BIAS_RESIDUAL(T)                                                                               \
  template void InvokeAddBiasResidual<T>(const void* input_a, const void* input_b, const void* bias, const int m, \
                                         const int n, void* output, cudaStream_t stream)
INVOKE_ADD_BIAS_RESIDUAL(float);
INVOKE_ADD_BIAS_RESIDUAL(half);
#ifdef ENABLE_BFLOAT16
INVOKE_ADD_BIAS_RESIDUAL(__nv_bfloat16);
#endif
#undef INVOKE_ADD_BIAS_RESIDUAL

template <typename T>
void InvokeSiluActivation(const void* input, const void* gated_weights, const int m, const int n, void* output,
                          cudaStream_t stream) {
  const int* ia3_tasks = nullptr;
  const T* bias = nullptr;
  const T* ia3_weights = nullptr;
  const T* gated_bias = nullptr;
  const int int8_mode = 0;
  const int* padding_offset = nullptr;
  const int seq_len = 0;
  const float* activation_in = nullptr;
  const float* activation_out = nullptr;
  CUDA_CHECK(cudaMemcpyAsync(output, input, sizeof(T) * m * n, cudaMemcpyDeviceToDevice, stream));
  llm_kernels::nvidia::InvokeGenericActivation<llm_kernels::nvidia::SiluActivation, T, T>(
      reinterpret_cast<T*>(output), bias, reinterpret_cast<const T*>(gated_weights), gated_bias, ia3_tasks, ia3_weights,
      m, n, int8_mode, activation_in, activation_out, padding_offset, seq_len, stream);
}

#define INVOKE_SILU_ACTIVATION(T)                                                                               \
  template void InvokeSiluActivation<T>(const void* input, const void* gated_weights, const int m, const int n, \
                                        void* output, cudaStream_t stream)
INVOKE_SILU_ACTIVATION(float);
INVOKE_SILU_ACTIVATION(half);
#ifdef ENABLE_BFLOAT16
INVOKE_SILU_ACTIVATION(__nv_bfloat16);
#endif
#undef INVOKE_SILU_ACTIVATION

template <typename T>
torch::ScalarType GetTorchDataType();
#define GET_TORCH_DATA_TYPE(T, TORCH_TYPE)  \
  template <>                               \
  torch::ScalarType GetTorchDataType<T>() { \
    return TORCH_TYPE;                      \
  }
GET_TORCH_DATA_TYPE(float, torch::kFloat32);
GET_TORCH_DATA_TYPE(half, torch::kFloat16);
#ifdef ENABLE_BFLOAT16
GET_TORCH_DATA_TYPE(__nv_bfloat16, torch::kBFloat16);
#endif
#undef GET_TORCH_DATA_TYPE

template <typename T>
void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* out, void* seqlen,
                 llm_kernels::nvidia::RotaryEmbeddingCuda<T>& rotary_embedding_cuda, int total_tokens, int max_tokens,
                 int batch, int num_heads, int head_size, int stride_size, int tensor_para_size, bool is_causal,
                 int rank, int block_size, void** k_list, void** v_list, void* block_offset,
                 const std::optional<void*>& alibi_slopes, cudaStream_t stream) {
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>());
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
                                   reinterpret_cast<T*>(q_tensor.data_ptr()), reinterpret_cast<T*>(k_tensor.data_ptr()),
                                   total_tokens, stream);
    rotary_embedding_cuda.Forward();
  }

  llm_kernels::nvidia::CacheCopy<T>(reinterpret_cast<T*>(k_tensor.data_ptr()),
                                    reinterpret_cast<T*>(v_tensor.data_ptr()), k_list, v_list,
                                    reinterpret_cast<size_t*>(seqlen), reinterpret_cast<int*>(block_offset), block_size,
                                    batch, total_tokens, num_heads, head_size, stride_size, stream);

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

#define ATTEN_VARLEN(T)                                                                                                \
  template void AttenVarlen<T>(                                                                                        \
      void* qkv_ptr, void* rotary_embedding_pos, void* out, void* seqlen,                                              \
      llm_kernels::nvidia::RotaryEmbeddingCuda<T>& rotary_embedding_cuda, int total_tokens, int max_tokens, int batch, \
      int num_heads, int head_size, int stride_size, int tensor_para_size, bool is_causal, int rank, int block_size,   \
      void** k_list, void** v_list, void* block_offset, const std::optional<void*>& alibi_slopes, cudaStream_t stream)
ATTEN_VARLEN(float);
ATTEN_VARLEN(half);
#ifdef ENABLE_BFLOAT16
ATTEN_VARLEN(__nv_bfloat16);
#endif
#undef ATTEN_VARLEN

template <typename T>
void PagedAttention(int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size, void* out,
                    void* q_tensor_ptr, void* key_cache_ptrs, void* value_cache_ptrs, void* cache_offsets_ptr,
                    void* context_lens_ptr, int max_context_len, int num_seqs, cudaStream_t& stream, void* workspace,
                    size_t work_size, const float* alibi_slopes_ptr);

#define PAGED_ATTENTION(T1, T2)                                                                                       \
  template <>                                                                                                         \
  void PagedAttention<T1>(int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size, void* out, \
                          void* q_tensor_ptr, void* key_cache_ptrs, void* value_cache_ptrs, void* cache_offsets_ptr,  \
                          void* context_lens_ptr, int max_context_len, int num_seqs, cudaStream_t& stream,            \
                          void* workspace, size_t work_size, const float* alibi_slopes_ptr) {                         \
    llm_kernels::nvidia::PagedAttentionCuda<T2> op;                                                                   \
    op.SetConfig(num_kv_heads, num_heads, head_size, block_size, stride_size);                                        \
    op.SetInput(reinterpret_cast<T2*>(out), reinterpret_cast<const T2*>(q_tensor_ptr),                                \
                reinterpret_cast<T2**>(key_cache_ptrs), reinterpret_cast<T2**>(value_cache_ptrs),                     \
                reinterpret_cast<const int*>(cache_offsets_ptr), reinterpret_cast<const int*>(context_lens_ptr),      \
                max_context_len, num_seqs, stream, workspace, work_size, alibi_slopes_ptr);                           \
    op.Forward();                                                                                                     \
  }
PAGED_ATTENTION(float, float);
PAGED_ATTENTION(half, uint16_t);
#ifdef ENABLE_BFLOAT16
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16);
#endif
#undef PAGED_ATTENTION

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
                         llm_kernels::nvidia::RotaryEmbeddingCuda<T>& rotary_embedding_cuda, void* workspace,
                         size_t work_size, int rank, const std::optional<void*>& alibi_slopes, void* qkv_workspace) {
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>());
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
    rotary_embedding_cuda.SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<T*>(q_tensor_ptr),
                                   reinterpret_cast<T*>(k_tensor_ptr), total_tokens, stream);
    rotary_embedding_cuda.Forward();
  }

  llm_kernels::nvidia::CachePosCopy<T>(
      reinterpret_cast<T*>(k_tensor_ptr), reinterpret_cast<T*>(v_tensor_ptr), key_cache_ptrs, value_cache_ptrs,
      rotary_embedding_pos, reinterpret_cast<size_t*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr),
      block_size, batch, total_tokens, num_heads, head_size, stride_size, stream);

  PagedAttention<T>(num_heads, head_size, num_kv_heads, stride_size, block_size, out, q_tensor_ptr, key_cache_ptrs,
                    value_cache_ptrs, cache_offsets_ptr, context_lens_ptr, max_context_len, num_seqs, stream, workspace,
                    work_size, alibi_slopes_ptr);
}

#define RUN_PAGED_ATTENTION(T)                                                                                         \
  template void run_paged_attention<T>(                                                                                \
      void* out, void* query, void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr,                  \
      int max_context_len, cudaStream_t stream, void* cache_offsets_ptr, int num_seqs, int num_heads, int head_size,   \
      int num_kv_heads, int stride_size, int block_size, int batch, void* rotary_embedding_pos, int total_tokens,      \
      llm_kernels::nvidia::RotaryEmbeddingCuda<T>& rotary_embedding_cuda, void* workspace, size_t work_size, int rank, \
      const std::optional<void*>& alibi_slopes, void* qkv_workspace)
RUN_PAGED_ATTENTION(float);
RUN_PAGED_ATTENTION(half);
#ifdef ENABLE_BFLOAT16
RUN_PAGED_ATTENTION(__nv_bfloat16);
#endif
#undef RUN_PAGED_ATTENTION

template <typename T>
void AssembleLastToken(const void* input, const void* offset, const int batch_size, const int hidden_units_num,
                       void* output, cudaStream_t& stream) {
  llm_kernels::nvidia::AssembleLastToken<T>(reinterpret_cast<const T*>(input), reinterpret_cast<const size_t*>(offset),
                                            batch_size, hidden_units_num, reinterpret_cast<T*>(output), stream);
}

#define ASSEMBEL_LAST_TOKEN(T)                                                                    \
  template void AssembleLastToken<T>(const void* input, const void* offset, const int batch_size, \
                                     const int hidden_units_num, void* output, cudaStream_t& stream);
ASSEMBEL_LAST_TOKEN(float);
ASSEMBEL_LAST_TOKEN(half);
#ifdef ENABLE_BFLOAT16
ASSEMBEL_LAST_TOKEN(__nv_bfloat16);
#endif
#undef ASSEMBEL_LAST_TOKEN

template <typename T>
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

#define CUSTOM_ALL_REDUCE_INIT(T)                                                                                   \
  template void CustomAllReduceInit<T>(void** ptr, void* input, void** metas, void* rank_data, void** data_handles, \
                                       void** input_handles, int data_size, size_t rank_data_sz, int tp_size,       \
                                       int rank, cudaStream_t& stream);
CUSTOM_ALL_REDUCE_INIT(float);
CUSTOM_ALL_REDUCE_INIT(half);
#ifdef ENABLE_BFLOAT16
CUSTOM_ALL_REDUCE_INIT(__nv_bfloat16);
#endif
#undef CUSTOM_ALL_REDUCE_INIT

template <typename T>
void CustomAllReduceRun(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream) {
  llm_kernels::nvidia::CustomAllreduce* reduce_op = static_cast<llm_kernels::nvidia::CustomAllreduce*>(ptr);
  reduce_op->AllReduce<T>(stream, static_cast<T*>(input), static_cast<T*>(result), data_size);
}

template void CustomAllReduceRun<float>(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);
template void CustomAllReduceRun<half>(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);
#ifdef ENABLE_BFLOAT16
template void CustomAllReduceRun<__nv_bfloat16>(void* ptr, void* input, void* result, int data_size,
                                                cudaStream_t& stream);
#endif

template <>
ncclDataType_t GetNcclDataType<float>() {
  return ncclFloat;
}
template <>
ncclDataType_t GetNcclDataType<half>() {
  return ncclHalf;
}
#ifdef ENABLE_BFLOAT16
template <>
ncclDataType_t GetNcclDataType<__nv_bfloat16>() {
  return ncclBfloat16;
}
#endif

template <typename T>
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
  llm_kernels::nvidia::InvokePermute<4ul, sizeof(T)>(input, output, input_shape, permutation, stream);
}
#define INVOKE_PERMUTE(T)                                                                    \
  template void InvokePermute<T>(void* input, void* output, std::vector<size_t> input_shape, \
                                 std::vector<size_t> permutation, cudaStream_t& stream);
INVOKE_PERMUTE(float);
INVOKE_PERMUTE(half);
#ifdef ENABLE_BFLOAT16
INVOKE_PERMUTE(__nv_bfloat16);
#endif
#undef INVOKE_PERMUTE

template <>
void DataToFloat<float>(const void* input, const int data_size, void* output, cudaStream_t& stream) {
  if (input != output) {
    cudaMemcpyAsync(output, input, data_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  }
}
template <>
void DataToFloat<half>(const void* input, const int data_size, void* output, cudaStream_t& stream) {
  llm_kernels::nvidia::HalfToFloat(reinterpret_cast<const half*>(input), data_size, reinterpret_cast<float*>(output),
                                   stream);
}
#ifdef ENABLE_BFLOAT16
template <>
void DataToFloat<__nv_bfloat16>(const void* input, const int data_size, void* output, cudaStream_t& stream) {
  llm_kernels::nvidia::BFloat16ToFloat(reinterpret_cast<const __nv_bfloat16*>(input), data_size,
                                       reinterpret_cast<float*>(output), stream);
}
#endif

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  if (tensor.dtype == DataType::TYPE_BF16 && target_dtype == DataType::TYPE_FP16) {
#ifdef ENABLE_BFLOAT16
    llm_kernels::nvidia::BFP16ToFP16(tensor.GetPtr<void>(), tensor.GetElementNumber(), stream.Get());
#endif
  } else if (tensor.dtype == DataType::TYPE_FP16 && target_dtype == DataType::TYPE_BF16) {
#ifdef ENABLE_BFLOAT16
    llm_kernels::nvidia::FP16ToBFP16(tensor.GetPtr<void>(), tensor.GetElementNumber(), stream.Get());
#endif
  } else if (tensor.dtype == target_dtype) {
    // No need to convert
  } else {
    throw std::runtime_error(
        fmt::format("CastInplace from type {} to {} is not yet implement", tensor.dtype, target_dtype));
  }
  tensor.dtype = DataType::TYPE_FP16;
  return Status();
}

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* workspace_ptr) {
  if (input_tensor.dtype == TYPE_FP32) {
    InvokePermute<float>(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape, permutation,
                         stream.Get());
  } else if (input_tensor.dtype == TYPE_FP16) {
    InvokePermute<half>(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape, permutation,
                        stream.Get());
  } else if (input_tensor.dtype == TYPE_BF16) {
#ifdef ENABLE_BFLOAT16
    InvokePermute<__nv_bfloat16>(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape,
                                 permutation, stream.Get());
#else
    throw std::runtime_error(fmt::format("Permute of type {} is not yet implement", input_tensor.dtype));
#endif
  } else {
    throw std::runtime_error(fmt::format("Permute of type {} is not yet implement", input_tensor.dtype));
  }
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
