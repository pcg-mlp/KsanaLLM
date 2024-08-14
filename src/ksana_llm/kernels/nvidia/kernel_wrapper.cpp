/* Copyright 2024 Tencent Inc.  All rights reserved.

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
#include "csrc/kernels/nvidia/samplers/greedy.h"

#include "ksana_llm/kernels/argmax.h"
#include "ksana_llm/kernels/cast.h"

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"

namespace ksana_llm {

template <typename T, llm_kernels::nvidia::WeightType WT>
void GetFpAIntBGroupGemmWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes) {
  auto gemm = llm_kernels::nvidia::FpAIntBGroupGemmWrapper<T, WT>();
  gemm.GetWorkspaceSize(m, n, k, ws_bytes);
}
#define GET_FPA_INTB_CONFIG(T, WT) \
  template void GetFpAIntBGroupGemmWorkspaceSize<T, WT>(size_t m, size_t n, size_t k, size_t & ws_bytes)
GET_FPA_INTB_CONFIG(float, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_CONFIG(float, llm_kernels::nvidia::WeightType::INT8);
GET_FPA_INTB_CONFIG(half, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_CONFIG(half, llm_kernels::nvidia::WeightType::INT8);
#ifdef ENABLE_BFLOAT16
GET_FPA_INTB_CONFIG(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_CONFIG(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
#endif
#undef GET_FPA_INTB_CONFIG

template <typename T, llm_kernels::nvidia::WeightType WT>
void InvokeFpAIntBGroupGemm(void* output, const void* input, const void* weight, const void* scales, void* ws, size_t m,
                            size_t n, size_t k, size_t groupsize, size_t config_index, cudaStream_t stream) {
  auto gemm = llm_kernels::nvidia::FpAIntBGroupGemmWrapper<T, WT>();
  gemm.Gemm(output, input, weight, scales, ws, m, n, k, groupsize, config_index, stream);
}
#define INVOKE_FPA_INTB_GROUP_GEMM(T, WT)                                                                              \
  template void InvokeFpAIntBGroupGemm<T, WT>(void* output, const void* input, const void* weight, const void* scales, \
                                              void* ws, size_t m, size_t n, size_t k, size_t groupsize,                \
                                              size_t config_index, cudaStream_t stream)
INVOKE_FPA_INTB_GROUP_GEMM(float, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_GEMM(float, llm_kernels::nvidia::WeightType::INT8);
INVOKE_FPA_INTB_GROUP_GEMM(half, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_GEMM(half, llm_kernels::nvidia::WeightType::INT8);
#ifdef ENABLE_BFLOAT16
INVOKE_FPA_INTB_GROUP_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
#endif
#undef INVOKE_FPA_INTB_GROUP_GEMM

template <typename T, llm_kernels::nvidia::WeightType WT>
size_t InvokeFpAIntBGroupGemmConfigProfile(size_t warmup, size_t iter, void* output, const void* input,
                                           const void* weight, const void* scales, void* ws, size_t m, size_t n,
                                           size_t k, size_t groupsize, cudaStream_t stream) {
  auto gemm = llm_kernels::nvidia::FpAIntBGroupGemmWrapper<T, WT>();
  return gemm.GetBestConfigIndex(warmup, iter, output, input, weight, scales, ws, m, n, k, groupsize, stream);
}
#define INVOKE_FPA_INTB_GROUP_GEMM_CONFIG_PROGILE(T, WT)                                                             \
  template size_t InvokeFpAIntBGroupGemmConfigProfile<T, WT>(                                                        \
      size_t warmup, size_t iter, void* output, const void* input, const void* weight, const void* scales, void* ws, \
      size_t m, size_t n, size_t k, size_t groupsize, cudaStream_t stream)
INVOKE_FPA_INTB_GROUP_GEMM_CONFIG_PROGILE(float, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_GEMM_CONFIG_PROGILE(float, llm_kernels::nvidia::WeightType::INT8);
INVOKE_FPA_INTB_GROUP_GEMM_CONFIG_PROGILE(half, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_GEMM_CONFIG_PROGILE(half, llm_kernels::nvidia::WeightType::INT8);
#ifdef ENABLE_BFLOAT16
INVOKE_FPA_INTB_GROUP_GEMM_CONFIG_PROGILE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_GEMM_CONFIG_PROGILE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
#endif
#undef INVOKE_FPA_INTB_GROUP_GEMM_CONFIG_PROGILE

template <typename T>
void LookupEmbedding(const void* ids, const void* offsets, const void* prefix_offsets, const void* emb, const void* pos,
                     void* output, int vocab_size, int hidden_size, int bs, int step, int vocab_id, cudaStream_t stream,
                     void* workspace_ptr) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), reinterpret_cast<const T*>(pos), {},
      reinterpret_cast<const int32_t*>(ids), step, reinterpret_cast<const size_t*>(offsets),
      reinterpret_cast<const size_t*>(prefix_offsets), bs, hidden_size, vocab_size, vocab_id, stream));
}
#define LOOKUP_EMBEDDING(T)                                                                                           \
  template void LookupEmbedding<T>(const void* ids, const void* offsets, const void* prefix_offsets, const void* emb, \
                                   const void* pos, void* output, int vocab_size, int hidden_size, int bs, int step,  \
                                   int vocab_id, cudaStream_t stream, void* workspace_ptr)
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
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeLayerNorm<T>(reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input),
                                              reinterpret_cast<const T*>(weight), beta, layernorm_eps, m, n, stream));
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
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeAddBiasResidual<T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input_a), reinterpret_cast<const T*>(input_b), nullptr,
      reinterpret_cast<const T*>(bias), nullptr, nullptr, m, n, stream));
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
  if (output != input) {
    throw std::runtime_error("Activation is an in-place operation, `output` must be the same as `input`.");
  }
  const int* ia3_tasks = nullptr;
  const T* bias = nullptr;
  const T* ia3_weights = nullptr;
  const T* gated_bias = nullptr;
  const int int8_mode = 0;
  const int* padding_offsets = nullptr;
  const int seq_len = 0;
  const float* activation_in = nullptr;
  const float* activation_out = nullptr;
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeGenericActivation<llm_kernels::nvidia::SiluActivation, T, T>(
      reinterpret_cast<T*>(output), bias, reinterpret_cast<const T*>(gated_weights), gated_bias, ia3_tasks, ia3_weights,
      m, n, int8_mode, activation_in, activation_out, padding_offsets, seq_len, stream));
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

// Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
// intended for use in testing accuracy outcomes only.
static bool kContextDecodeUseFP8Cache = []() -> bool {
  const char* val = std::getenv("ContextDecodeUseFP8Cache");
  if (val != nullptr) {
    return true;
  }
  return false;
}();

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen,
                 llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>& rotary_embedding_cuda, int total_tokens,
                 int max_tokens, int batch, int num_heads, int num_kv_heads, int head_size, int stride_size,
                 float k_scale, float v_scale, int tensor_para_size, bool is_causal, int rank, int block_size,
                 void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
                 const std::optional<void*>& alibi_slopes, cudaStream_t stream) {
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  torch::Tensor qkv_tensor =
      torch::from_blob(qkv_ptr, {total_tokens, (num_heads + num_kv_heads * 2) * head_size}, options);
  auto tt = qkv_tensor.split({num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size}, -1);
  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);
  torch::Tensor seqlen_tensor = torch::from_blob(seqlen, {batch + 1}, int_options);

  // rotary embedding
  torch::Tensor q_tensor = tt[0];
  torch::Tensor k_tensor = tt[1];
  torch::Tensor v_tensor = tt[2];

  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ReverseCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
      v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(prefix_offsets),
      reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
      k_scale, v_scale, stream));

  if (!alibi_slopes.has_value()) {
    rotary_embedding_cuda.SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                   reinterpret_cast<int64_t*>(rotary_embedding_mask),
                                   reinterpret_cast<SCALAR_T*>(q_tensor.data_ptr()),
                                   reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda.Forward());
  }

  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
      v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(prefix_offsets),
      reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
      k_scale, v_scale, stream));

// flash attention 2 or flash attention 1
#ifdef ENABLE_FLASH_ATTN_2
  // refer to github Dao-AILab/flash-attention csrc/flash_attn/flash_api.cpp#L374
  // When the flag is set to True and the output is not nullptr, calling the function mha_varlen_fwd
  // leads to a core dump.
  bool seqlenq_ngroups_swapped =
      max_tokens == 1 && num_heads > num_kv_heads && head_size % 8 == 0 && !alibi_slopes.has_value();
  c10::optional<at::Tensor> out_tensor = c10::nullopt;
  if (!seqlenq_ngroups_swapped) {
    out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);
  }
  at::Tensor q_tmp_tensor = torch::reshape(q_tensor, {total_tokens, num_heads, head_size});
  c10::optional<at::Tensor> seqused_k = c10::nullopt;
  auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
  c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
  if (alibi_slopes.has_value()) {
    alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
  }
  // Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
  // intended for use in testing accuracy outcomes only.
  if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kAuto) {
    if (kContextDecodeUseFP8Cache) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), k_tensor.size(0), k_tensor.size(1), stride_size, k_scale,
          stream));
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), v_tensor.size(0), v_tensor.size(1), stride_size, v_scale,
          stream));
    }
  }
  std::vector<at::Tensor> mha_output =
      mha_varlen_fwd(q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                     torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor,
                     seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), seqused_k, alibi_slopes_tensor,
                     max_tokens, max_tokens, 0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, false, c10::nullopt);
  if (seqlenq_ngroups_swapped) {
    KLLM_LOG_DEBUG << "To prevent a core dump when seqlenq_ngroups_swapped is True, set the output tensor to nullptr.";
    at::Tensor& out_data = mha_output[0];
    size_t total_size = out_data.numel() * out_data.element_size();
    CUDA_CHECK(cudaMemcpyAsync(out, out_data.data_ptr(), total_size, cudaMemcpyDeviceToDevice, stream));
  }
#else
  c10::optional<at::Tensor> out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);
  flash_attn::mha_varlen_fwd(torch::reshape(q_tensor, {total_tokens, num_heads, head_size}),
                             torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                             torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor,
                             seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), max_tokens, max_tokens,
                             0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, false, c10::nullopt);
#endif
}

#define ATTEN_VARLEN(SCALAR_T, CACHE_T, KV_DTYPE)                                                                  \
  template void AttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(                                                          \
      void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen,             \
      llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>& rotary_embedding_cuda, int total_tokens, int max_tokens, \
      int batch, int num_heads, int num_kv_heads, int head_size, int stride_size, float k_scale, float v_scale,    \
      int tensor_para_size, bool is_causal, int rank, int block_size, void** k_list, void** v_list,                \
      void* prefix_offsets, void* block_offsets, const std::optional<void*>& alibi_slopes, cudaStream_t stream)
ATTEN_VARLEN(float, float, llm_kernels::utils::KVCacheType::kAuto);
ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
ATTEN_VARLEN(half, half, llm_kernels::utils::KVCacheType::kAuto);
ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#ifdef ENABLE_BFLOAT16
ATTEN_VARLEN(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef ATTEN_VARLEN

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void PagedAttention(int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale,
                    float v_scale, void* out, void* q_tensor_ptr, void* key_cache_ptrs, void* value_cache_ptrs,
                    void* cache_offsets_ptr, void* context_lens_ptr, int max_context_len, int num_seqs,
                    cudaStream_t& stream, void* workspace, size_t work_size, const float* alibi_slopes_ptr);

#define PAGED_ATTENTION(T1, T2, CACHE_T1, CACHE_T2, KV_DTYPE)                                                        \
  template <>                                                                                                        \
  void PagedAttention<T1, CACHE_T1, KV_DTYPE>(                                                                       \
      int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale, float v_scale, \
      void* out, void* q_tensor_ptr, void* key_cache_ptrs, void* value_cache_ptrs, void* cache_offsets_ptr,          \
      void* context_lens_ptr, int max_context_len, int num_seqs, cudaStream_t& stream, void* workspace,              \
      size_t work_size, const float* alibi_slopes_ptr) {                                                             \
    llm_kernels::nvidia::PagedAttentionCuda<T2, CACHE_T2, KV_DTYPE> op;                                              \
    op.SetConfig(num_kv_heads, num_heads, head_size, block_size, stride_size, k_scale, v_scale);                     \
    op.SetInput(reinterpret_cast<T2*>(out), reinterpret_cast<const T2*>(q_tensor_ptr),                               \
                reinterpret_cast<CACHE_T2**>(key_cache_ptrs), reinterpret_cast<CACHE_T2**>(value_cache_ptrs),        \
                reinterpret_cast<const int*>(cache_offsets_ptr), reinterpret_cast<const int*>(context_lens_ptr),     \
                max_context_len, num_seqs, stream, workspace, work_size, alibi_slopes_ptr);                          \
    CUDA_CHECK_LAST_ERROR(op.Forward());                                                                             \
  }
PAGED_ATTENTION(float, float, float, float, llm_kernels::utils::KVCacheType::kAuto);
PAGED_ATTENTION(float, float, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
PAGED_ATTENTION(float, float, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
PAGED_ATTENTION(half, uint16_t, half, uint16_t, llm_kernels::utils::KVCacheType::kAuto);
PAGED_ATTENTION(half, uint16_t, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
PAGED_ATTENTION(half, uint16_t, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#ifdef ENABLE_BFLOAT16
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef PAGED_ATTENTION

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokePagedAttention(void* output_ptr, void* query_ptr, void** key_cache_ptrs, void** value_cache_ptrs,
                          void* context_lens_ptr, int max_context_len, cudaStream_t stream, void* cache_offsets_ptr,
                          int seqs_num, int heads_num, int head_size, int kv_heads_num, int stride_size, int block_size,
                          float k_scale, float v_scale, int batch, void* rotary_embedding_pos,
                          void* rotary_embedding_mask, int total_tokens,
                          llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>& rotary_embedding_cuda,
                          void* workspace_ptr, size_t work_size, int rank, const std::optional<void*>& alibi_slopes,
                          void* qkv_workspace) {
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  torch::Tensor qkv_tensor =
      torch::from_blob(query_ptr, {total_tokens, (heads_num + kv_heads_num * 2) * head_size}, options);
  auto tt = qkv_tensor.split({heads_num * head_size, kv_heads_num * head_size, kv_heads_num * head_size}, -1);

  torch::Tensor q_tensor = tt[0];
  torch::Tensor k_tensor = tt[1];
  torch::Tensor v_tensor = tt[2];
  void* q_tensor_ptr = q_tensor.data_ptr();
  void* k_tensor_ptr = k_tensor.data_ptr();
  void* v_tensor_ptr = v_tensor.data_ptr();

  if (!alibi_slopes.has_value()) {
    // When the alibi_slopes parameter is empty, execute the rotary embedding.
    rotary_embedding_cuda.SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_tensor_ptr), reinterpret_cast<SCALAR_T*>(k_tensor_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda.Forward());
  }

  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(v_tensor_ptr), key_cache_ptrs,
      value_cache_ptrs, rotary_embedding_pos, reinterpret_cast<size_t*>(context_lens_ptr),
      reinterpret_cast<int*>(cache_offsets_ptr), block_size, batch, total_tokens, kv_heads_num, head_size, stride_size,
      k_scale, v_scale, stream));

  PagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(heads_num, head_size, kv_heads_num, stride_size, block_size, k_scale,
                                              v_scale, output_ptr, q_tensor_ptr, key_cache_ptrs, value_cache_ptrs,
                                              cache_offsets_ptr, context_lens_ptr, max_context_len, seqs_num, stream,
                                              workspace_ptr, work_size, alibi_slopes_ptr);
}

#define RUN_PAGED_ATTENTION(SCALAR_T, CACHE_T, KV_DTYPE)                                                             \
  template void InvokePagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(                                                   \
      void* output_ptr, void* query_ptr, void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr,     \
      int max_context_len, cudaStream_t stream, void* cache_offsets_ptr, int seqs_num, int heads_num, int head_size, \
      int kv_heads_num, int stride_size, int block_size, float k_scale, float v_scale, int batch,                    \
      void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens,                                     \
      llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>& rotary_embedding_cuda, void* workspace_ptr,                \
      size_t work_size, int rank, const std::optional<void*>& alibi_slopes, void* qkv_workspace)
RUN_PAGED_ATTENTION(float, float, llm_kernels::utils::KVCacheType::kAuto);
RUN_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_PAGED_ATTENTION(half, half, llm_kernels::utils::KVCacheType::kAuto);
RUN_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#ifdef ENABLE_BFLOAT16
RUN_PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
RUN_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef RUN_PAGED_ATTENTION

template <typename T>
void AssembleLastToken(const void* inputs, const void* offsets, const void* prefix_offsets, const int batch_size,
                       const int hidden_units_num, void* output, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::AssembleLastToken<T>(
      reinterpret_cast<const T*>(inputs), reinterpret_cast<const size_t*>(offsets),
      reinterpret_cast<const size_t*>(prefix_offsets), batch_size, hidden_units_num, reinterpret_cast<T*>(output),
      stream));
}

#define ASSEMBEL_LAST_TOKEN(T)                                                                            \
  template void AssembleLastToken<T>(const void* inputs, const void* offsets, const void* prefix_offsets, \
                                     const int batch_size, const int hidden_units_num, void* output,      \
                                     cudaStream_t& stream);
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
    KLLM_LOG_ERROR << "input != input_handles[rank]";
  }
  std::vector<std::string> handles;
  handles.reserve(tp_size);
  for (int i = 0; i < tp_size; i++) {
    char* begin = reinterpret_cast<char*>(&input_handles[i]);
    char* end = reinterpret_cast<char*>(&input_handles[i + 1]);
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
  KLLM_CHECK_WITH_INFO(input_shape.size() <= 4ul,
                       fmt::format("input shape dims number {} > 4 is not supported", input_shape.size()));
  if (input_shape.empty()) {
    return;
  }

  // Extend to num_dims = 4
  input_shape.resize(4, 1);
  for (size_t i = permutation.size(); i < 4; ++i) {
    permutation.push_back(i);
  }
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokePermute<4ul, sizeof(T)>(input, output, input_shape, permutation, stream));
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
void DataToFloat<float>(const void* input, const int data_size, const size_t vocab_size, const size_t vocab_size_pad,
                        void* output, cudaStream_t& stream) {
  if (input != output) {
    if (vocab_size != vocab_size_pad) {
      // It should be implemented when supporting float inference.
      KLLM_LOG_ERROR << "Float to float does not support Stride.";
    }
    CUDA_CHECK(cudaMemcpyAsync(output, input, data_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  }
}
template <>
void DataToFloat<half>(const void* input, const int data_size, const size_t vocab_size, const size_t vocab_size_pad,
                       void* output, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToFloat(reinterpret_cast<const half*>(input), data_size,
                                                         reinterpret_cast<float*>(output), stream, vocab_size_pad,
                                                         vocab_size));
}
#ifdef ENABLE_BFLOAT16
template <>
void DataToFloat<__nv_bfloat16>(const void* input, const int data_size, const size_t vocab_size,
                                const size_t vocab_size_pad, void* output, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToFloat(reinterpret_cast<const __nv_bfloat16*>(input), data_size,
                                                             reinterpret_cast<float*>(output), stream, vocab_size_pad,
                                                             vocab_size));
}
#endif

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  if (tensor.dtype == DataType::TYPE_BF16 && target_dtype == DataType::TYPE_FP16) {
#ifdef ENABLE_BFLOAT16
    CUDA_CHECK_LAST_ERROR(
        llm_kernels::nvidia::BFP16ToFP16(tensor.GetPtr<void>(), tensor.GetElementNumber(), stream.Get()));
#endif
  } else if (tensor.dtype == DataType::TYPE_FP16 && target_dtype == DataType::TYPE_BF16) {
#ifdef ENABLE_BFLOAT16
    CUDA_CHECK_LAST_ERROR(
        llm_kernels::nvidia::FP16ToBFP16(tensor.GetPtr<void>(), tensor.GetElementNumber(), stream.Get()));
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

void CalcLogprobs(float* logits, float* temperatures, int vocab_size, int bs, int logprobs_num, float* logprobs,
                  int64_t* token_ids) {
  auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat32);
  auto logits_tensor = torch::from_blob(logits, {bs, vocab_size}, options);
  auto temperatures_tensor = torch::from_blob(temperatures, {bs}, options);

  torch::Tensor logits_sort, logits_idx;
  std::tie(logits_sort, logits_idx) = logits_tensor.sort(-1, true);
  logits_sort = logits_sort.narrow(1, 0, logprobs_num)
                    .div_(temperatures_tensor.unsqueeze_(1))
                    .log_softmax(-1)
                    .to(torch::kCPU)
                    .view({-1});
  logits_idx = logits_idx.narrow(1, 0, logprobs_num).to(torch::kCPU).view({-1});

  memcpy(logprobs, logits_sort.data_ptr<float>(), logprobs_num * bs * sizeof(float));
  memcpy(token_ids, logits_idx.data_ptr<int64_t>(), logprobs_num * bs * sizeof(int64_t));
}

template <typename T>
Status ArgMax(const T* input, const uint32_t* ids_offsets, const int32_t batch_size, const int32_t vocab_size,
              uint32_t* result, Stream& stream, void* workspace_ptr) {
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeArgMaxReduce(input, ids_offsets, batch_size, vocab_size, result, stream.Get()));
  return Status();
}

#define INSTANTIATE_ARG_MAX(T)                                                                  \
  template Status ArgMax(const T* input, const uint32_t* ids_offsets, const int32_t batch_size, \
                         const int32_t vocab_size, uint32_t* result, Stream& stream, void* workspace_ptr);

INSTANTIATE_ARG_MAX(float);
INSTANTIATE_ARG_MAX(half);
#ifdef ENABLE_BF16
INSTANTIATE_ARG_MAX(__nv_bfloat16);
#endif

#undef INSTANTIATE_ARG_MAX

#ifdef ENABLE_FP8
#  define INSTANTIATE_FP8_DYNAMIC_QUANTIZE(T)                                                                        \
    template <>                                                                                                      \
    void Fp8DynamicQuantize<T>(int num_channels, int channel_size, const T* input_ptr, void* quant_ptr,              \
                               float* scale_ptr, cudaStream_t& stream) {                                             \
      CUDA_CHECK_LAST_ERROR(llm_kernels::utils::InvokeComputeFP8QuantizeScale<T>(scale_ptr, input_ptr, num_channels, \
                                                                                 channel_size, stream));             \
      CUDA_CHECK_LAST_ERROR(llm_kernels::utils::InvokeQuantizeMatrix<__nv_fp8_e4m3, T>(                              \
          static_cast<__nv_fp8_e4m3*>(quant_ptr), scale_ptr, input_ptr, num_channels, channel_size, stream));        \
    }
INSTANTIATE_FP8_DYNAMIC_QUANTIZE(float);
INSTANTIATE_FP8_DYNAMIC_QUANTIZE(half);
#  ifdef ENABLE_BFLOAT16
INSTANTIATE_FP8_DYNAMIC_QUANTIZE(__nv_bfloat16);
#  endif
#  undef INSTANTIATE_FP8_DYNAMIC_QUANTIZE

#  define INVOKE_FP8_QUANTIZED_MATMUL(T, CUDA_TYPE)                                                                 \
    template <>                                                                                                     \
    void Fp8QuantizedMatMul<T>(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k, \
                               const void* a_ptr, const void* a_scale, const void* b_ptr, const void* b_scale,      \
                               T* c_ptr, cudaStream_t& stream) {                                                    \
      CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(                                                             \
          cublas_handle, cublaslt_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, b_ptr, k, CUDA_R_8F_E4M3, a_ptr, k,    \
          CUDA_R_8F_E4M3, c_ptr, n, CUDA_TYPE, 1.0f, 0.f, CUDA_R_32F, stream, nullptr, nullptr, a_scale, b_scale)); \
    }

INVOKE_FP8_QUANTIZED_MATMUL(float, CUDA_R_32F);
INVOKE_FP8_QUANTIZED_MATMUL(half, CUDA_R_16F);
#  ifdef ENABLE_BFLOAT16
INVOKE_FP8_QUANTIZED_MATMUL(__nv_bfloat16, CUDA_R_16BF);
#  endif
#  undef INVOKE_FP8_QUANTIZED_MATMUL
#endif
}  // namespace ksana_llm
