/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/flash_mla_attention_layer.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "csrc/kernels/nvidia/blockwise_gemm/blockwise_gemm.h"
#include "csrc/kernels/nvidia/others/sglang/main/quantization/fp8/per_token_group_quant.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy_flash_attn_layout.h"
#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "csrc/utils/nvidia/cuda_fp8_utils.h"
#include "device/nvidia/cuda_utils.h"
#include "executor/layer_progress_tracker.h"
#include "kernels/nvidia/basic_kernel_wrapper.h"
#include "kernels/nvidia/flash_attn_cpp_wrapper.h"
#include "pd_v2/prefill/pd_v2_layer_event_tracker.h"
#include "utils/logger.h"
#include "utils/search_status.h"
#include "utils/singleton.h"

namespace ksana_llm {

extern bool kContextDecodeUseFP8Cache;

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaAttenVarlenAbsorb(void* output_buffer, void* q_nope_rope_ptr, void* k_pe_ptr, void* compressed_kv_ptr,
                          void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale,
                          void* v_head_weight_scale, void* layer_workspace, cublasHandle_t& cublas_handles,
                          cublasLtHandle_t& cublaslt_handles, void* k_buffer, void* v_buffer,
                          size_t* seqlens_with_prefix_ptr, float attn_scale,
                          std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda,
                          int max_tokens, int max_kv_tokens, int batch_size, int num_heads, int qk_rope_head_dim,
                          int qk_nope_head_dim, int kv_lora_rank, int v_head_dim, float k_scale, float v_scale,
                          void* k_scale_ptr, void* v_scale_ptr, bool is_causal, int rank, int block_size, void** k_list,
                          size_t* prefix_offsets, int* block_offsets, cudaStream_t stream, int total_prefix_tokens,
                          size_t* seqlens_without_prefix_ptr, void* seqlens_without_prefix_int32_ptr,
                          QuantMode mm_quant_mode, void* src_flexible_rotary_embedding_pos_ptr,
                          void* dst_flexible_rotary_embedding_pos_ptr, void* flexible_rotary_embedding_mask_ptr,
                          void* flexible_offset_uint64_ptr, int flexible_len, size_t chunk_q_tokens_offset,
                          size_t chunk_q_tokens, size_t chunk_kv_tokens_offset, size_t chunk_kv_tokens,
                          int chunk_batch_size, void* chunk_kv_cu_seq_lens_ptr) {
  // output_buffer layout(temporary use): [k_nope]
  void* k_nope_ptr = static_cast<SCALAR_T*>(output_buffer) + chunk_q_tokens_offset * num_heads * qk_nope_head_dim;
  void* k_rope_buffer = k_pe_ptr;
  void* k_ptr = k_buffer;
  void* v_ptr = v_buffer;

  // TODO(yfnjin): When prefix caching is enabled, keep this section of code here due to dependency of
  // rotary embedding/kv cache
  if (total_prefix_tokens > 0) {
    const size_t k_nope_size = chunk_kv_tokens * num_heads * qk_nope_head_dim * sizeof(SCALAR_T);
    const size_t k_rope_size = chunk_kv_tokens * qk_rope_head_dim * sizeof(SCALAR_T);
    const size_t latent_size = chunk_kv_tokens * kv_lora_rank * sizeof(SCALAR_T);
    const size_t k_size = k_nope_size + k_rope_size * num_heads;
    k_nope_ptr = layer_workspace;
    k_rope_buffer = k_nope_ptr + k_nope_size;
    void* latent_buffer = k_rope_buffer + k_rope_size;
    k_ptr = latent_buffer;
    v_ptr = k_ptr + k_size;

    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
      // get latent and k_rope from cache block (include prefix and new)
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaGetFromCompressedCache<SCALAR_T, CACHE_T, KV_DTYPE>(
          k_rope_buffer, latent_buffer, k_list, chunk_kv_tokens, seqlens_with_prefix_ptr, block_offsets, block_size,
          qk_rope_head_dim, kv_lora_rank, stream, chunk_kv_tokens_offset));
    } else {
      if (chunk_q_tokens < chunk_kv_tokens) {
        // If prefix length > 0, we holds a new buffer to store total kv (include prefix and new).
        // For fp8 kv cache, we dequantize the prefix part from cache block, and copy the new part from existing buffer.
        // Dequantize and copy the prefix part from cache block.
        CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaFlashPrefixKVReverseCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
            reinterpret_cast<SCALAR_T*>(k_rope_buffer), reinterpret_cast<SCALAR_T*>(latent_buffer), k_list,
            prefix_offsets, seqlens_with_prefix_ptr, block_offsets, block_size, chunk_kv_tokens, qk_rope_head_dim,
            kv_lora_rank, k_scale, v_scale, stream, chunk_kv_tokens_offset));
      }
      // Copy the new (without prefix) part from existing buffer to avoid loss of percision.
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaFlashWithoutPrefixKVCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_rope_buffer), reinterpret_cast<SCALAR_T*>(latent_buffer),
          reinterpret_cast<SCALAR_T*>(k_pe_ptr), reinterpret_cast<SCALAR_T*>(compressed_kv_ptr), prefix_offsets,
          seqlens_without_prefix_ptr, chunk_q_tokens, qk_rope_head_dim, kv_lora_rank, stream, chunk_q_tokens_offset,
          chunk_kv_tokens_offset));
    }

    if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kFp8DsMla) {
      if (flexible_len != 0) {
        // TODO(zhongzhicao): For DeepSeek model, the twice rope of flexible cached tokens can be fused by calculating
        // the subtraction of dst_token_idx and src_token_idx. However, in subsequent development, the token_idx in
        // src_req and dst_req may not guarantee consistency, thus there could be both positive and negative value in
        // flexible_rotary_embedding_pos, requiring an additional tensor to mark whether to excute reverse rope.
        if (rotary_embedding_cuda.has_value()) {
          // reverse rope for flexible cached tokens, with is_reverse flag setting to true.
          rotary_embedding_cuda->SetInput(
              reinterpret_cast<int64_t*>(src_flexible_rotary_embedding_pos_ptr) + chunk_kv_tokens_offset,
              reinterpret_cast<int64_t*>(flexible_rotary_embedding_mask_ptr) + chunk_kv_tokens_offset, nullptr,
              k_rope_buffer, chunk_kv_tokens, stream, 0, /*key_stride=*/0, 0, qk_rope_head_dim,
              /* is_reverse */ true);
          CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward<SCALAR_T>());

          // correct rope for flexible cached tokens
          rotary_embedding_cuda->SetInput(
              reinterpret_cast<int64_t*>(dst_flexible_rotary_embedding_pos_ptr) + chunk_kv_tokens_offset,
              reinterpret_cast<int64_t*>(flexible_rotary_embedding_mask_ptr) + chunk_kv_tokens_offset, nullptr,
              k_rope_buffer, chunk_kv_tokens, stream, 0, /*key_stride=*/0, 0, qk_rope_head_dim);
          CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward<SCALAR_T>());
        }

        // copy flexible cached k to k cache block
        CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaFlashFlexibleKCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
            reinterpret_cast<SCALAR_T*>(k_rope_buffer), k_list, reinterpret_cast<size_t*>(flexible_offset_uint64_ptr),
            prefix_offsets, seqlens_with_prefix_ptr, block_offsets, block_size, batch_size, chunk_kv_tokens,
            qk_rope_head_dim, kv_lora_rank, k_scale, stream, chunk_kv_tokens_offset));
      }
    }

    // calc k_nope by latent_buffer @ kv_b_nope_proj. k_nope: [token_num, head, qk_nope_head_dim]
    if (kv_b_nope_weight_scale != nullptr) {
      if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
        SCALAR_T* a = static_cast<SCALAR_T*>(latent_buffer);
        void* a_q = latent_buffer + latent_size;
        float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * chunk_kv_tokens * kv_lora_rank);
        InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, chunk_kv_tokens, kv_lora_rank, true, stream);
        float* b_scale = static_cast<float*>(kv_b_nope_weight_scale);
        InvokeBlockGemm<SCALAR_T>(a_q, a_s, kv_b_nope_proj_weight, b_scale, k_nope_ptr, chunk_kv_tokens, kv_lora_rank,
                                  num_heads * qk_nope_head_dim, stream);
      } else if (mm_quant_mode == QUANT_GPTQ) {
        KLLM_CHECK_WITH_INFO(layer_workspace != nullptr, "Quantized matmul has not layer_workspace");
        int64_t workspace_size = 0;
        std::vector<std::string> machete_schedule_map =
            Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * qk_nope_head_dim,
                                                                              kv_lora_rank);
        std::optional<std::string> best_schedule = std::nullopt;
        if (static_cast<size_t>(chunk_kv_tokens) < machete_schedule_map.size()) {
          best_schedule = std::optional<std::string>(machete_schedule_map[chunk_kv_tokens]);
        }
        InvokeMacheteGemm(workspace_size, layer_workspace, stream, chunk_kv_tokens, num_heads * qk_nope_head_dim,
                          kv_lora_rank, latent_buffer, kv_b_nope_proj_weight, k_nope_ptr,
                          GetMacheteDataType<SCALAR_T>(), llm_kernels::nvidia::vllm_dtype::kU4B8,
                          kv_b_nope_weight_scale,
                          std::optional<std::vector<size_t>>({static_cast<size_t>(kv_lora_rank / 128),
                                                              static_cast<size_t>(num_heads * qk_nope_head_dim)}),
                          GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
      }
    } else {
      InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, chunk_kv_tokens, num_heads * qk_nope_head_dim,
                             kv_lora_rank, reinterpret_cast<const void*>(latent_buffer),
                             reinterpret_cast<const void*>(kv_b_nope_proj_weight), k_nope_ptr, stream, nullptr,
                             nullptr);
    }

    // calc value by latent_buffer @ v_head_proj. value: [token_num, head, v_head_dim]
    if (v_head_weight_scale != nullptr) {
      if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
        void* a_q = latent_buffer + latent_size;
        float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * chunk_kv_tokens * kv_lora_rank);
        float* b_scale = static_cast<float*>(v_head_weight_scale);
        InvokeBlockGemm<SCALAR_T>(a_q, a_s, v_head_proj_weight, b_scale, v_ptr, chunk_kv_tokens, kv_lora_rank,
                                  num_heads * v_head_dim, stream);
      } else if (mm_quant_mode == QUANT_GPTQ) {
        int64_t workspace_size = 0;
        std::vector<std::string> machete_schedule_map =
            Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * v_head_dim, kv_lora_rank);
        std::optional<std::string> best_schedule = std::nullopt;
        if (static_cast<size_t>(chunk_kv_tokens) < machete_schedule_map.size()) {
          best_schedule = std::optional<std::string>(machete_schedule_map[chunk_kv_tokens]);
        }
        InvokeMacheteGemm(workspace_size, layer_workspace, stream, chunk_kv_tokens, num_heads * v_head_dim,
                          kv_lora_rank, latent_buffer, v_head_proj_weight, v_ptr, GetMacheteDataType<SCALAR_T>(),
                          llm_kernels::nvidia::vllm_dtype::kU4B8, v_head_weight_scale,
                          std::optional<std::vector<size_t>>(
                              {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * v_head_dim)}),
                          GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
      }
    } else {
      InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, chunk_kv_tokens, num_heads * v_head_dim, kv_lora_rank,
                             reinterpret_cast<const void*>(latent_buffer),
                             reinterpret_cast<const void*>(v_head_proj_weight), v_ptr, stream, nullptr, nullptr);
    }
  }

  // cat(k_nope, k_pe)
  ConcatMlaK<SCALAR_T>(k_nope_ptr, k_rope_buffer, k_ptr, chunk_kv_tokens, num_heads, qk_nope_head_dim, qk_rope_head_dim,
                       stream);

  const int qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
  // Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
  // intended for use in testing accuracy outcomes only.
  if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kAuto &&
                KV_DTYPE != llm_kernels::utils::KVCacheType::kFp8DsMla) {
    if (kContextDecodeUseFP8Cache) {
      const int stride_size_k = num_heads * qk_head_dim;
      const int stride_size_v = num_heads * v_head_dim;
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_ptr), chunk_kv_tokens, num_heads, stride_size_k, k_scale, stream));
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(v_ptr), chunk_kv_tokens, num_heads, stride_size_v, v_scale, stream));
    }
  }

  // FA3 handles variable dimensions natively, no padding needed
  // Initialize torch options of input and output
  const auto output_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  auto input_options = output_options;

  // Only use FP8 inference for FA3
  // FA3 is only applicable to Hopper architecture, so it is unnecessary to check the compute capability
  c10::optional<at::Tensor> k_descale = c10::nullopt;
  c10::optional<at::Tensor> v_descale = c10::nullopt;
  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    void* const quant_k_ptr = layer_workspace;  // reuse k_nope_ptr for quantized k tensor
    // TODO(zhongzhicao): Quantization of K tensor can be fused into Concat
    llm_kernels::nvidia::ConvertToCacheType<SCALAR_T, CACHE_T, KV_DTYPE>(
        /*k_src*/ reinterpret_cast<SCALAR_T*>(k_ptr), /*k_dst*/ reinterpret_cast<CACHE_T*>(quant_k_ptr),
        chunk_kv_tokens, num_heads, qk_head_dim, num_heads * qk_head_dim, k_scale, stream);
    void* const quant_v_ptr = k_ptr;  // reuse v_ptr memory for quantized v tensor
    llm_kernels::nvidia::ConvertToCacheType<SCALAR_T, CACHE_T, KV_DTYPE>(
        /*v_src*/ reinterpret_cast<SCALAR_T*>(v_ptr), /*v_dst*/ reinterpret_cast<CACHE_T*>(quant_v_ptr),
        chunk_kv_tokens, num_heads, v_head_dim, num_heads * v_head_dim, v_scale, stream);

    // Set inpt torch options for FP8
    input_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat8_e4m3fn);
    k_ptr = quant_k_ptr;
    v_ptr = quant_v_ptr;

    if (k_scale_ptr != nullptr) {
      const auto scale_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
      k_descale = torch::from_blob(k_scale_ptr, {1, num_heads}, scale_options);
      v_descale = torch::from_blob(v_scale_ptr, {1, num_heads}, scale_options);
    }
  }

  torch::Tensor q_tensor =
      torch::from_blob(q_nope_rope_ptr, {static_cast<int64_t>(chunk_q_tokens), num_heads, qk_head_dim}, input_options);
  torch::Tensor k_tensor =
      torch::from_blob(k_ptr, {static_cast<int64_t>(chunk_kv_tokens), num_heads, qk_head_dim}, input_options);
  torch::Tensor v_tensor =
      torch::from_blob(v_ptr, {static_cast<int64_t>(chunk_kv_tokens), num_heads, v_head_dim}, input_options);
  // refer to github Dao-AILab/flash-attention csrc/flash_attn/flash_api.cpp#L374
  // When the flag is set to True and the output is not nullptr, calling the function mha_varlen_fwd
  // leads to a core dump.
  c10::optional<at::Tensor> out_tensor =
      torch::from_blob(static_cast<SCALAR_T*>(output_buffer) + chunk_q_tokens_offset * num_heads * v_head_dim,
                       {static_cast<int64_t>(chunk_q_tokens), num_heads, v_head_dim}, output_options);

  const auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt32);
  const torch::Tensor seqlen_q_tensor =
      torch::from_blob(seqlens_without_prefix_int32_ptr, {chunk_batch_size + 1}, int32_options);
  const torch::Tensor seqlen_kv_tensor =
      torch::from_blob(chunk_kv_cu_seq_lens_ptr, {chunk_batch_size + 1}, int32_options);
  c10::optional<at::Tensor> seqused_k = c10::nullopt;
  c10::optional<at::Tensor> block_table = c10::nullopt;

  std::vector<at::Tensor> mha_output;
  {
    MhaVarlenFwdParams params;
    params.q = q_tensor;
    params.k = k_tensor;
    params.v = v_tensor;
    params.out = out_tensor;
    params.seqlen_q = seqlen_q_tensor;
    params.seqlen_k = seqlen_kv_tensor;
    params.seqused_k = seqused_k;
    params.max_seqlen_q = max_tokens;
    params.max_seqlen_k = max_kv_tokens;
    params.block_table = block_table;
    params.p_dropout = 0.f;
    params.softmax_scale = static_cast<double>(attn_scale);
    params.zero_tensors = false;
    params.is_causal = is_causal;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0f;
    params.return_softmax = false;
    params.gen = c10::nullopt;
    params.q_descale = k_descale;
    params.k_descale = k_descale;
    params.v_descale = v_descale;
    mha_output = InvokeMhaVarlenFwd(params);
  }
}

#define MLA_ATTEN_VARLEN_ABSORB(SCALAR_T, CACHE_T, KV_DTYPE)                                                          \
  template void MlaAttenVarlenAbsorb<SCALAR_T, CACHE_T, KV_DTYPE>(                                                    \
      void* output_buffer, void* q_nope_rope_ptr, void* k_pe_ptr, void* compressed_kv_ptr,                            \
      void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale, void* v_head_weight_scale, \
      void* layer_workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles, void* k_buffer,      \
      void* v_buffer, size_t* seqlens_with_prefix_ptr, float attn_scale,                                              \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda, int max_tokens,                 \
      int max_kv_tokens, int batch_size, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank, \
      int v_head_dim, float k_scale, float v_scale, void* k_scale_ptr, void* v_scale_ptr, bool is_causal, int rank,   \
      int block_size, void** k_list, size_t* prefix_offsets, int* block_offsets, cudaStream_t stream,                 \
      int total_prefix_tokens, size_t* seqlens_without_prefix_ptr, void* seqlens_without_prefix_int32_ptr,            \
      QuantMode mm_quant_mode, void* src_flexible_rotary_embedding_pos_ptr,                                           \
      void* dst_flexible_rotary_embedding_pos_ptr, void* flexible_rotary_embedding_mask_ptr,                          \
      void* flexible_offset_uint64_ptr, int flexible_len, size_t chunk_q_tokens_offset, size_t chunk_q_tokens,        \
      size_t chunk_kv_tokens_offset, size_t chunk_kv_tokens, int chunk_batch_size, void* chunk_kv_cu_seq_lens_ptr)
MLA_ATTEN_VARLEN_ABSORB(float, float, llm_kernels::utils::KVCacheType::kAuto);
MLA_ATTEN_VARLEN_ABSORB(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN_ABSORB(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_ATTEN_VARLEN_ABSORB(half, half, llm_kernels::utils::KVCacheType::kAuto);
MLA_ATTEN_VARLEN_ABSORB(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN_ABSORB(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_ATTEN_VARLEN_ABSORB(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
#if defined(ENABLE_FP8)
MLA_ATTEN_VARLEN_ABSORB(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN_ABSORB(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef MLA_ATTEN_VARLEN_ABSORB

Status FlashMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                    std::shared_ptr<Context> context, int rank) {
#ifndef ENABLE_FLASH_ATTN_WITH_CACHE
  KLLM_THROW("MLA Only support ENABLE_FLASH_ATTN_WITH_CACHE.");
#endif
  if (!IsUsingFA3()) {
    KLLM_THROW("MLA只支持FA3，请在配置中启用FlashAttention 3");
  }
  max_step_token_num_ = runtime_config.max_step_token_num;
  max_chunked_kv_tokens_ = runtime_config.max_chunked_kv_tokens;
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

size_t FlashMlaAttentionLayer::GetWorkspaceSize() {
  size_t workspace_size_ = 0;
  if (kv_cache_dtype_ == TYPE_FP8_E4M3) {
    // quant_q_ptr
    const size_t workspace_size_per_token =
        this->num_heads_ * (this->qk_rope_head_dim_ + this->qk_nope_head_dim_) * GetTypeSize(kv_cache_dtype_);
    // k_scale, v_scale
    workspace_size_ = max_step_token_num_ * workspace_size_per_token + 2 * this->num_heads_ * sizeof(float);
  }
  // k_nope_ptr/quant_k_ptr, k_rope_ptr, latent_buffer/k_ptr/quant_v_ptr, v_ptr
  size_t workspace_size_per_token =
      (this->num_heads_ * (2 * qk_nope_head_dim_ + qk_rope_head_dim_ + v_head_dim_) + qk_rope_head_dim_) *
      GetTypeSize(inter_data_type_);
  // Workspace must cover the largest possible chunk, which is bounded by max_chunked_kv_tokens_.
  workspace_size_ += max_chunked_kv_tokens_ * workspace_size_per_token;

  KLLM_LOG_DEBUG << fmt::format("FlashMlaAttentionLayer requires workspace: {}", workspace_size_);
  return workspace_size_;
}

Status FlashMlaAttentionLayer::Forward(const FlashMlaAttentionInputs& inputs, Tensor* output) {
  if (kv_cache_dtype_ == TYPE_FP8_DS_MLA) {
    switch (inter_data_type_) {
      case DataType::TYPE_FP16:
        return ForwardT<float16, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla>(inputs, output);
      case DataType::TYPE_BF16:
        return ForwardT<bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla>(inputs, output);
      default:
        KLLM_THROW(fmt::format("{}: Unsupported Dtype type: {}.", __PRETTY_FUNCTION__, inter_data_type_));
    }
  }
  DISPATCH_BY_DTYPE_AND_KVTYPE(inter_data_type_, kv_cache_dtype_, ForwardT, inputs, output);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashMlaAttentionLayer::ForwardT(const FlashMlaAttentionInputs& inputs, Tensor* out) {
  const Tensor& dp_input_offset = *inputs.dp_input_offset;
  const size_t* dp_input_offset_uint64_ptr = inputs.dp_input_offset_host->GetPtr<size_t>();
  const Tensor& kv_list = *inputs.kv_list;
  const Tensor& dp_input_prefix = *inputs.dp_input_prefix;
  const Tensor& dp_prefill_q_offset = *inputs.dp_prefill_q_offset;
  const size_t* dp_prefill_q_offset_uint64_ptr = inputs.dp_prefill_q_offset_host->GetPtr<size_t>();
  const Tensor& chunk_kv_batch_starts_host = *inputs.chunk_kv_batch_starts_host;
  const Tensor& chunk_kv_cu_extend_lens_tensor = *inputs.chunk_kv_cu_extend_lens;
  const Tensor& chunk_kv_cu_seq_lens_tensor = *inputs.chunk_kv_cu_seq_lens;
  const Tensor& kv_cache_offset = *inputs.kv_cache_offset;
  const Tensor& rotary_embedding_pos =
      *inputs.rotary_embedding_pos;  // rope for q/k [total_q_tokens, num_q_heads/1, head_size]
  const Tensor& rotary_embedding_mask =
      *inputs.rotary_embedding_mask;  // mask for q/k [total_q_tokens, num_q_heads/1, head_size]
  const Tensor& dp_src_flexible_rotary_embedding_pos =
      *inputs.dp_src_flexible_rotary_embedding_pos;  // reverse rope for k [total_tokens, head_size]
  const Tensor& dp_dst_flexible_rotary_embedding_pos =
      *inputs.dp_dst_flexible_rotary_embedding_pos;  // correct rope for k [total_tokens, head_size]
  const Tensor& dp_flexible_rotary_embedding_mask =
      *inputs.dp_flexible_rotary_embedding_mask;  // mask for k [total_tokens, head_size]
  const Tensor& dp_dst_flexible_kv_cache = *inputs.dp_dst_flexible_kv_cache;
  const Tensor& dp_src_flexible_kv_cache = *inputs.dp_src_flexible_kv_cache;
  const Tensor& dp_dst_flexible_token_idx = *inputs.dp_dst_flexible_token_idx;
  const Tensor& dp_src_flexible_token_idx = *inputs.dp_src_flexible_token_idx;
  const Tensor& dp_flexible_offset_uint64 = *inputs.dp_flexible_offset_uint64;
  const Tensor& forward_shape = *inputs.forward_shape;
  const Tensor& q_nope_rope_tensor = *inputs.q_nope_rope;
  const Tensor& kv_buffer = *inputs.kv_buffer;
  const Tensor& k_rope_buffer = *inputs.k_rope_buffer;
  const Tensor& kv_b_nope_proj_weight = *inputs.kv_b_nope_proj_weight;
  const Tensor& v_head_proj_weight = *inputs.v_head_proj_weight;
  const Tensor& k_buffer = *inputs.k_buffer;
  const Tensor& v_buffer = *inputs.v_buffer;

  Tensor& output = *out;

  const int layer_block_num = forward_shape.shape[2];
  const int max_tokens = forward_shape.shape[6];
  const int batch_size = forward_shape.shape[7];
  const int max_kv_tokens = forward_shape.shape[8];
  const int total_prefix_tokens = forward_shape.shape[11];
  const int total_q_tokens = q_nope_rope_tensor.shape[0];

  void** const k_list = kv_list.GetPtr<void*>() + this->layer_index_ * layer_block_num * 2;

  void* kv_b_nope_weight_scale = nullptr;
  void* v_head_weight_scale = nullptr;
  if (this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3) {
    kv_b_nope_weight_scale = kv_b_nope_proj_weight.weight_scales->GetPtr<void>();
    v_head_weight_scale = v_head_proj_weight.weight_scales->GetPtr<void>();
  } else if (this->mm_quant_mode_ == QUANT_GPTQ) {
    kv_b_nope_weight_scale = kv_b_nope_proj_weight.scales->GetPtr<void>();
    v_head_weight_scale = v_head_proj_weight.scales->GetPtr<void>();
  }

  auto& stream = this->context_->device->GetPrimaryComputeStream();

  const int qk_head_dim = this->qk_nope_head_dim_ + this->qk_rope_head_dim_;
  const int flexible_len = dp_dst_flexible_kv_cache.shape[0];
  // Copy cache of flexible tokens from src blocks to dst blocks
  if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kFp8DsMla) {
    if (total_prefix_tokens > 0 && flexible_len != 0) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaFlexibleTokenCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
          dp_src_flexible_kv_cache.GetPtr<CACHE_T*>(), dp_dst_flexible_kv_cache.GetPtr<CACHE_T*>(),
          dp_src_flexible_token_idx.GetPtr<int>(), dp_dst_flexible_token_idx.GetPtr<int>(), this->block_token_num_,
          this->layer_index_, flexible_len, this->qk_rope_head_dim_ + this->kv_lora_rank_, stream.Get()));
    }
  }

  if (this->rotary_embedding_cuda_.has_value()) {
    this->rotary_embedding_cuda_->SetInput(
        rotary_embedding_pos.GetPtr<int64_t>(), rotary_embedding_mask.GetPtr<int64_t>(),
        q_nope_rope_tensor.GetPtr<SCALAR_T>() + this->qk_nope_head_dim_, k_rope_buffer.GetPtr<SCALAR_T>(),
        total_q_tokens, stream.Get(), this->num_heads_ * qk_head_dim,
        /*key_stride*/ 0, qk_head_dim, this->qk_rope_head_dim_);
    CUDA_CHECK_LAST_ERROR(this->rotary_embedding_cuda_->Forward<SCALAR_T>());
  }

  // copy new k&v to kv cache block
  // Use compressed kvcache, k is [num_token, qk_rope_head_dim], v is  [num_token, kv_lora_rank]
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaFlashKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      kv_buffer.GetPtr<SCALAR_T>(), k_rope_buffer.GetPtr<SCALAR_T>(), k_list, k_list, dp_input_prefix.GetPtr<size_t>(),
      dp_prefill_q_offset.GetPtr<size_t>(), kv_cache_offset.GetPtr<int>(), this->block_token_num_, batch_size,
      total_q_tokens, this->kv_lora_rank_, this->qk_rope_head_dim_, this->k_scale_, this->v_scale_, stream.Get()));

  void* q_nope_rope_ptr = q_nope_rope_tensor.GetPtr<void>();
  void* fp8_work_buffer = this->workspace_buffer_ ? this->workspace_buffer_->template GetPtr<void>() : nullptr;
  void* k_scale_ptr = nullptr;
  void* v_scale_ptr = nullptr;
  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2) {
    KLLM_THROW("Flash Attention 3 not support fp8_e5m2 KV Cache. Please use fp8_e4m3.");
  } else if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    if constexpr (!std::is_same<SCALAR_T, __nv_bfloat16>::value) {
      KLLM_THROW("Flash Attention 3 only supports BF16 output for FP8 input.");
    }
    // Quant input tensors to FP8 for E4M3 kv cache type
    const float q_scale = this->k_scale_;
    void* const quant_q_ptr = fp8_work_buffer;
    llm_kernels::nvidia::ConvertToCacheType<SCALAR_T, CACHE_T, KV_DTYPE>(
        /*q_src*/ reinterpret_cast<SCALAR_T*>(q_nope_rope_ptr),
        /*q_dst*/ reinterpret_cast<CACHE_T*>(quant_q_ptr), total_q_tokens, this->num_heads_, qk_head_dim,
        this->num_heads_ * qk_head_dim, q_scale, stream.Get());
    q_nope_rope_ptr = quant_q_ptr;
    const size_t quant_q_size = static_cast<size_t>(total_q_tokens) * this->num_heads_ * qk_head_dim * sizeof(CACHE_T);
    fp8_work_buffer += quant_q_size;

    // FA3 requires scale inputs in [batch_size = 1, num_heads]
    if (this->k_scale_ != 1.0f || this->v_scale_ != 1.0f) {
      const size_t scale_size = this->num_heads_ * sizeof(float);
      k_scale_ptr = fp8_work_buffer;
      v_scale_ptr = fp8_work_buffer + scale_size;
      fp8_work_buffer += 2 * scale_size;
      KLLM_LOG_DEBUG << "Valid kv scale detected, preparing FA3 scale inputs.";
      llm_kernels::nvidia::InvokeFillKVScaleIntoBuffer(k_scale_ptr, v_scale_ptr, &this->k_scale_, &this->v_scale_,
                                                       this->num_heads_, stream.Get());
    }
  }

  // Process KV in chunks
  int* chunk_kv_batch_starts_ptr = chunk_kv_batch_starts_host.GetPtr<int>();
  const size_t num_chunks = chunk_kv_batch_starts_host.shape[0] - 1;
  const size_t q_nope_rope_per_token_size =
      static_cast<size_t>(this->num_heads_) * qk_head_dim *
      (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8DsMla ? sizeof(SCALAR_T) : sizeof(CACHE_T));
  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int chunk_kv_batch_start = chunk_kv_batch_starts_ptr[chunk_idx];
    const int chunk_kv_batch_end = chunk_kv_batch_starts_ptr[chunk_idx + 1];
    const size_t chunk_q_tokens_offset = dp_prefill_q_offset_uint64_ptr[chunk_kv_batch_start];
    const size_t chunk_kv_tokens_offset = dp_input_offset_uint64_ptr[chunk_kv_batch_start];
    MlaAttenVarlenAbsorb<SCALAR_T, CACHE_T, KV_DTYPE>(
        output.GetPtr<void>(), q_nope_rope_ptr + chunk_q_tokens_offset * q_nope_rope_per_token_size,
        k_rope_buffer.GetPtr<SCALAR_T>() + chunk_q_tokens_offset * this->qk_rope_head_dim_,
        kv_buffer.GetPtr<SCALAR_T>() + chunk_q_tokens_offset * this->kv_lora_rank_,
        kv_b_nope_proj_weight.GetPtr<void>(), v_head_proj_weight.GetPtr<void>(), kv_b_nope_weight_scale,
        v_head_weight_scale, fp8_work_buffer, this->context_->device->ext->GetCublasHandle(),
        this->context_->device->ext->GetCublasLtHandle(), k_buffer.GetPtr<void>(),
        v_buffer.GetPtr<SCALAR_T>() + chunk_q_tokens_offset * this->num_heads_ * this->v_head_dim_,
        dp_input_offset.GetPtr<size_t>(), this->attn_scale_, this->rotary_embedding_cuda_, max_tokens, max_kv_tokens,
        batch_size, this->num_heads_, this->qk_rope_head_dim_, this->qk_nope_head_dim_, this->kv_lora_rank_,
        this->v_head_dim_, this->k_scale_, this->v_scale_, k_scale_ptr, v_scale_ptr, this->is_causal_, this->rank_,
        this->block_token_num_, k_list, dp_input_prefix.GetPtr<size_t>(), kv_cache_offset.GetPtr<int>(), stream.Get(),
        total_prefix_tokens, dp_prefill_q_offset.GetPtr<size_t>(),
        chunk_kv_cu_extend_lens_tensor.GetPtr<int>() + (chunk_kv_batch_start + chunk_idx), this->mm_quant_mode_,
        dp_src_flexible_rotary_embedding_pos.GetPtr<void>(), dp_dst_flexible_rotary_embedding_pos.GetPtr<void>(),
        dp_flexible_rotary_embedding_mask.GetPtr<void>(), dp_flexible_offset_uint64.GetPtr<void>(), flexible_len,
        chunk_q_tokens_offset, dp_prefill_q_offset_uint64_ptr[chunk_kv_batch_end] - chunk_q_tokens_offset,
        chunk_kv_tokens_offset, dp_input_offset_uint64_ptr[chunk_kv_batch_end] - chunk_kv_tokens_offset,
        chunk_kv_batch_end - chunk_kv_batch_start,
        chunk_kv_cu_seq_lens_tensor.GetPtr<int>() + (chunk_kv_batch_start + chunk_idx));
  }

  KLLM_LOG_DEBUG << "RecordLayerProgress, layer_index: " << this->layer_index_ << ", rank: " << this->rank_;
  // 通知 LayerProgressTracker 该层已完成，它会在内部记录 event 并在单独的线程中监控完成情况
  Singleton<LayerProgressTracker>::GetInstance()->RecordLayerProgress(this->rank_, this->layer_index_, stream);

  // pd_v2 Prefill: per-layer KV-write hook (parallel to V1 tracker; only
  // active when InferenceExecutor on Prefill role has Initialize'd the
  // PdV2 tracker — IsActive() short-circuits to false otherwise).
  auto pd_v2_tracker = Singleton<pd_v2::PdV2LayerEventTracker>::GetInstance();
  if (pd_v2_tracker->IsActive()) {
    pd_v2_tracker->RecordLayerProgress(this->rank_, this->layer_index_,
                                       this->context_->device->GetPrimaryComputeStream());
  }

  output.shape = {static_cast<size_t>(total_q_tokens), static_cast<size_t>(this->num_heads_ * this->v_head_dim_)};
  return Status();
}

}  // namespace ksana_llm
