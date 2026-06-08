/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/compressor_layer.h"

#include <fmt/core.h>

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/compressor_wrapper.h"
#include "layers/attention_layer.h"
#include "utils/logger.h"

namespace ksana_llm {

using llm_kernels::nvidia::CompressorWrapper;
using llm_kernels::nvidia::ForwardMode;
using llm_kernels::nvidia::PrefillPlan;
using Fp32Wrapper = CompressorWrapper<float, __nv_bfloat16>;
using Bf16Wrapper = CompressorWrapper<__nv_bfloat16, __nv_bfloat16>;

// Dispatch flash C4/C128 operations by compressor_dtype
#define DISPATCH_FLASH_BY_COMPRESSOR_DTYPE(compressor_dtype, ...)                                                   \
  switch (compressor_dtype) {                                                                                       \
    case TYPE_BF16:                                                                                                 \
      Bf16Wrapper::__VA_ARGS__;                                                                                     \
      break;                                                                                                        \
    case TYPE_FP32:                                                                                                 \
      Fp32Wrapper::__VA_ARGS__;                                                                                     \
      break;                                                                                                        \
    default:                                                                                                        \
      KLLM_THROW(fmt::format("CompressorLayer: unsupported compressor_dtype {}", GetTypeString(compressor_dtype))); \
  }
// Dispatch CompressFusedNormRopeStore by kv_cache_dtype
#define DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype, head_dim, ...)                                        \
  switch (kv_cache_dtype) {                                                                                     \
    case TYPE_FP8_DS_MLA:                                                                                       \
      Bf16Wrapper::template CompressFusedNormRopeStore<head_dim, llm_kernels::utils::KVCacheType::kFp8DsMla>(   \
          __VA_ARGS__);                                                                                         \
      break;                                                                                                    \
    case TYPE_FP8_E4M3:                                                                                         \
      Bf16Wrapper::template CompressFusedNormRopeStore<head_dim, llm_kernels::utils::KVCacheType::kFp8E4M3>(    \
          __VA_ARGS__);                                                                                         \
      break;                                                                                                    \
    default:                                                                                                    \
      KLLM_THROW(fmt::format("CompressorLayer: unsupported kv_cache_dtype {}", GetTypeString(kv_cache_dtype))); \
  }

// =============================================================================
// Init
// =============================================================================
Status CompressorLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  KLLM_CHECK_WITH_INFO(inter_data_type_ == DataType::TYPE_BF16,
                       fmt::format("CompressorLayer: unsupported inter_data_type {}", GetTypeString(inter_data_type_)));

  int param_idx = 0;
  compress_ratio_ = std::any_cast<uint32_t>(parameters[param_idx++]);
  int n_heads = std::any_cast<int>(parameters[param_idx++]);
  head_dim_ = std::any_cast<int>(parameters[param_idx++]);
  int rope_head_dim = std::any_cast<int>(parameters[param_idx++]);
  rms_norm_eps_ = std::any_cast<float>(parameters[param_idx++]);
  layer_idx_ = std::any_cast<int>(parameters[param_idx++]);

  is_overlap_ = (compress_ratio_ == 4);

  KLLM_CHECK(compress_ratio_ == 4 || compress_ratio_ == 128);
  KLLM_CHECK(head_dim_ == 128 || head_dim_ == 512);
  if (compress_ratio_ == 128) {
    KLLM_CHECK(head_dim_ == 512);
  }

  // Initialize freqs_cis for RoPE
  PositionEncoding position_encoding = std::any_cast<PositionEncoding>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(position_encoding == PositionEncoding::ROPE,
                       fmt::format("CompressorLayer: unsupported position encoding {}", position_encoding));

  float compress_rope_theta = std::any_cast<float>(parameters[param_idx++]);
  freqs_cis_ = std::any_cast<void*>(parameters[param_idx++]);
  auto rope_scaling_factor_config = std::any_cast<RoPEScalingFactor>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(rope_scaling_factor_config.type == "yarn",
                       fmt::format("CompressorLayer: unsupported rope type {}", rope_scaling_factor_config.type));
  const int max_position_embeddings = std::any_cast<int>(parameters[param_idx++]);

  // Parse fused-store parameters
  const int block_token_num = std::any_cast<int>(parameters[param_idx++]);
  const uint32_t compressed_block_token_num = static_cast<uint32_t>(block_token_num) / compress_ratio_;
  KLLM_CHECK(compressed_block_token_num > 0 && (compressed_block_token_num & (compressed_block_token_num - 1)) == 0);
  compressed_page_bits_ = __builtin_ctz(compressed_block_token_num);
  KLLM_CHECK_WITH_INFO(compressed_page_bits_ == 1 || compressed_page_bits_ == 6 || compressed_page_bits_ == 8,
                       "CompressorLayer: compressed_page_bits must be in {1, 6, 8}");
  page_stride_ = std::any_cast<int64_t>(parameters[param_idx++]);
  kv_cache_dtype_ = std::any_cast<DataType>(parameters[param_idx++]);
  compressor_dtype_ = std::any_cast<DataType>(parameters[param_idx++]);

  DISPATCH_BY_2_DTYPE(inter_data_type_, InitYarnRotaryEmbedding, rotary_embedding_cuda_, rope_scaling_factor_config,
                      freqs_cis_, compress_rope_theta, rope_head_dim, max_position_embeddings, head_dim_, n_heads,
                      /*is_neox*/ false, context_->device->GetPrimaryComputeStream().Get());
}

// =============================================================================
// Prefill – C4 (compress_ratio = 4, overlap mode)
// =============================================================================
Status CompressorLayer::ForwardPrefillC4(const Tensor* kv_score_buffer_tensor, const Tensor* kv_score_input,
                                         const Tensor* score_bias, const Tensor* indices, const Tensor* extra,
                                         Tensor* compress_plan, Tensor* write_plan, const Tensor* seq_lens,
                                         const Tensor* extend_lens, const Tensor* norm_weight,
                                         const Tensor* layer_kv_cache_ptr, const Tensor* compressed_out_loc,
                                         Tensor* output, bool plan_initialized) {
  KLLM_CHECK_WITH_INFO(kv_score_input->dtype == compressor_dtype_,
                       "kv_score_input dtype must be aligned with compressor dtype");
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  void* kv_score_buffer = kv_score_buffer_tensor->GetPtr<void*>()[layer_idx_];
  const void* kv_score_input_ptr = kv_score_input->GetPtr<void>();
  const void* score_bias_ptr = score_bias->GetPtr<void>();
  const int32_t* indices_ptr = indices->GetPtr<int32_t>();
  const void* extra_ptr = extra->GetPtr<void>();
  auto* compress_plan_ptr = compress_plan->GetPtr<PrefillPlan>();
  auto* write_plan_ptr = write_plan->GetPtr<PrefillPlan>();
  const int32_t* seq_lens_ptr = seq_lens->GetPtr<int32_t>();
  const int32_t* extend_lens_ptr = extend_lens->GetPtr<int32_t>();
  void* kv_compressed_output = output->GetPtr<void>();
  const uint32_t num_tokens = static_cast<uint32_t>(kv_score_input->shape[0]);

  // Step 1: build plan if needed
  if (!plan_initialized) {
    const uint32_t batch_size = static_cast<uint32_t>(indices->shape[0]);
    DISPATCH_FLASH_BY_COMPRESSOR_DTYPE(
        compressor_dtype_, PlanPrefillCompress(compress_plan_ptr, write_plan_ptr, seq_lens_ptr, extend_lens_ptr,
                                               batch_size, num_tokens, compress_ratio_, is_overlap_, stream));
  }

  // Step 2: prefill compression
  const uint32_t num_compress = static_cast<uint32_t>(compress_plan->shape[0]);
  const uint32_t num_write = static_cast<uint32_t>(write_plan->shape[0]);
  if (head_dim_ == 512) {
    DISPATCH_FLASH_BY_COMPRESSOR_DTYPE(
        compressor_dtype_, template FlashC4Prefill<512>(kv_score_buffer, kv_score_input_ptr, kv_compressed_output,
                                                        score_bias_ptr, indices_ptr, compress_plan_ptr, write_plan_ptr,
                                                        extra_ptr, num_compress, num_write, stream));
  } else {
    DISPATCH_FLASH_BY_COMPRESSOR_DTYPE(
        compressor_dtype_, template FlashC4Prefill<128>(kv_score_buffer, kv_score_input_ptr, kv_compressed_output,
                                                        score_bias_ptr, indices_ptr, compress_plan_ptr, write_plan_ptr,
                                                        extra_ptr, num_compress, num_write, stream));
  }

  // Step 3: Fused RMSNorm + RoPE + FP8 quantize + store to paged cache
  const void* norm_weight_ptr = norm_weight->GetPtr<void>();
  void* cache = layer_kv_cache_ptr->GetPtr<void*>()[1 + layer_idx_ * 2];
  const int32_t* compressed_out_loc_ptr = compressed_out_loc->GetPtr<int32_t>();
  if (head_dim_ == 512) {
    DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype_, 512, kv_compressed_output, norm_weight_ptr, compress_plan_ptr,
                                      freqs_cis_, num_compress, ForwardMode::CompressExtend, rms_norm_eps_,
                                      compress_ratio_, cache, compressed_out_loc_ptr, page_stride_,
                                      compressed_page_bits_, stream);
  } else {
    Bf16Wrapper::template IndexerFusedNormRopeStore<128>(
        kv_compressed_output, norm_weight_ptr, compress_plan_ptr, freqs_cis_, num_compress, ForwardMode::CompressExtend,
        rms_norm_eps_, compress_ratio_, cache, compressed_out_loc_ptr, page_stride_, compressed_page_bits_, stream);
  }

  return Status();
}

// =============================================================================
// Prefill – C128 (compress_ratio = 128, non-overlap mode)
// =============================================================================
Status CompressorLayer::ForwardPrefillC128(const Tensor* kv_score_buffer_tensor, const Tensor* kv_score_input,
                                           const Tensor* score_bias, const Tensor* indices, const Tensor* load_indices,
                                           Tensor* compress_plan, Tensor* write_plan, const Tensor* seq_lens,
                                           const Tensor* extend_lens, const Tensor* norm_weight,
                                           const Tensor* layer_kv_cache_ptr, const Tensor* compressed_out_loc,
                                           Tensor* output, bool plan_initialized) {
  KLLM_CHECK_WITH_INFO(kv_score_input->dtype == compressor_dtype_,
                       "kv_score_input dtype must be aligned with compressor dtype");
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  void* kv_score_buffer = kv_score_buffer_tensor->GetPtr<void*>()[layer_idx_];
  const void* kv_score_input_ptr = kv_score_input->GetPtr<void>();
  const void* score_bias_ptr = score_bias->GetPtr<void>();
  const int32_t* indices_ptr = indices->GetPtr<int32_t>();
  const int32_t* load_indices_ptr = load_indices->GetPtr<int32_t>();
  auto* compress_plan_ptr = compress_plan->GetPtr<PrefillPlan>();
  auto* write_plan_ptr = write_plan->GetPtr<PrefillPlan>();
  const int32_t* seq_lens_ptr = seq_lens->GetPtr<int32_t>();
  const int32_t* extend_lens_ptr = extend_lens->GetPtr<int32_t>();
  void* kv_compressed_output = output->GetPtr<void>();
  const uint32_t num_tokens = static_cast<uint32_t>(kv_score_input->shape[0]);

  // Step 1: build plan if needed
  if (!plan_initialized) {
    const uint32_t batch_size = static_cast<uint32_t>(indices->shape[0]);
    DISPATCH_FLASH_BY_COMPRESSOR_DTYPE(
        compressor_dtype_, PlanPrefillCompress(compress_plan_ptr, write_plan_ptr, seq_lens_ptr, extend_lens_ptr,
                                               batch_size, num_tokens, compress_ratio_, is_overlap_, stream));
  }

  // Step 2: prefill compression (C128 always has head_dim=512)
  const uint32_t num_compress = static_cast<uint32_t>(compress_plan->shape[0]);
  const uint32_t num_write = static_cast<uint32_t>(write_plan->shape[0]);
  DISPATCH_FLASH_BY_COMPRESSOR_DTYPE(
      compressor_dtype_, template FlashC128Prefill<512>(
                             kv_score_buffer, kv_score_input_ptr, kv_compressed_output, score_bias_ptr, indices_ptr,
                             load_indices_ptr, compress_plan_ptr, write_plan_ptr, num_compress, num_write, stream));

  // Step 3: Fused RMSNorm + RoPE + FP8 quantize + store to paged cache (C128 always head_dim=512)
  const void* norm_weight_ptr = norm_weight->GetPtr<void>();
  void* cache = layer_kv_cache_ptr->GetPtr<void*>()[1 + layer_idx_ * 2];
  const int32_t* compressed_out_loc_ptr = compressed_out_loc->GetPtr<int32_t>();
  DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype_, 512, kv_compressed_output, norm_weight_ptr, compress_plan_ptr,
                                    freqs_cis_, num_compress, ForwardMode::CompressExtend, rms_norm_eps_,
                                    compress_ratio_, cache, compressed_out_loc_ptr, page_stride_, compressed_page_bits_,
                                    stream);

  return Status();
}

// =============================================================================
// Decode – C4
// =============================================================================
Status CompressorLayer::ForwardDecodeC4(const Tensor* kv_score_buffer_tensor, const Tensor* kv_score_input,
                                        const Tensor* score_bias, const Tensor* indices, const Tensor* seq_lens,
                                        const Tensor* extra, const Tensor* norm_weight,
                                        const Tensor* layer_kv_cache_ptr, const Tensor* compressed_out_loc,
                                        Tensor* output) {
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  void* kv_score_buffer = kv_score_buffer_tensor->GetPtr<void*>()[layer_idx_];
  const void* kv_score_input_ptr = kv_score_input->GetPtr<void>();
  const void* score_bias_ptr = score_bias->GetPtr<void>();
  const int32_t* indices_ptr = indices->GetPtr<int32_t>();
  const int32_t* seq_lens_ptr = seq_lens->GetPtr<int32_t>();
  const int32_t* extra_ptr = extra->GetPtr<int32_t>();
  void* kv_compressed_output = output->GetPtr<void>();

  // Step 1: decode compression
  const uint32_t batch_size = static_cast<uint32_t>(indices->shape[0]);
  if (head_dim_ == 512) {
    DISPATCH_FLASH_BY_COMPRESSOR_DTYPE(
        compressor_dtype_,
        template FlashC4Decode<512>(kv_score_buffer, kv_score_input_ptr, kv_compressed_output, score_bias_ptr,
                                    indices_ptr, seq_lens_ptr, extra_ptr, batch_size, stream));
  } else {
    DISPATCH_FLASH_BY_COMPRESSOR_DTYPE(
        compressor_dtype_,
        template FlashC4Decode<128>(kv_score_buffer, kv_score_input_ptr, kv_compressed_output, score_bias_ptr,
                                    indices_ptr, seq_lens_ptr, extra_ptr, batch_size, stream));
  }

  // Step 2: Fused RMSNorm + RoPE + FP8 quantize + store to paged cache
  const void* norm_weight_ptr = norm_weight->GetPtr<void>();
  void* cache = layer_kv_cache_ptr->GetPtr<void*>()[1 + layer_idx_ * 2];
  const int32_t* compressed_out_loc_ptr = compressed_out_loc->GetPtr<int32_t>();
  if (head_dim_ == 512) {
    DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype_, 512, kv_compressed_output, norm_weight_ptr, seq_lens_ptr,
                                      freqs_cis_, batch_size, ForwardMode::CompressDecode, rms_norm_eps_,
                                      compress_ratio_, cache, compressed_out_loc_ptr, page_stride_,
                                      compressed_page_bits_, stream);
  } else {
    Bf16Wrapper::template IndexerFusedNormRopeStore<128>(
        kv_compressed_output, norm_weight_ptr, seq_lens_ptr, freqs_cis_, batch_size, ForwardMode::CompressDecode,
        rms_norm_eps_, compress_ratio_, cache, compressed_out_loc_ptr, page_stride_, compressed_page_bits_, stream);
  }

  return Status();
}

// =============================================================================
// Decode – C128
// =============================================================================
Status CompressorLayer::ForwardDecodeC128(const Tensor* kv_score_buffer_tensor, const Tensor* kv_score_input,
                                          const Tensor* score_bias, const Tensor* indices, const Tensor* seq_lens,
                                          const Tensor* norm_weight, const Tensor* layer_kv_cache_ptr,
                                          const Tensor* compressed_out_loc, Tensor* output) {
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  void* kv_score_buffer = kv_score_buffer_tensor->GetPtr<void*>()[layer_idx_];
  const void* kv_score_input_ptr = kv_score_input->GetPtr<void>();
  const void* score_bias_ptr = score_bias->GetPtr<void>();
  const int32_t* indices_ptr = indices->GetPtr<int32_t>();
  const int32_t* seq_lens_ptr = seq_lens->GetPtr<int32_t>();
  void* kv_compressed_output = output->GetPtr<void>();

  // Step 1: decode compression (C128 always has head_dim=512)
  const uint32_t batch_size = static_cast<uint32_t>(indices->shape[0]);
  DISPATCH_FLASH_BY_COMPRESSOR_DTYPE(
      compressor_dtype_, template FlashC128Decode<512>(kv_score_buffer, kv_score_input_ptr, kv_compressed_output,
                                                       score_bias_ptr, indices_ptr, seq_lens_ptr, batch_size, stream));

  // Step 2: Fused RMSNorm + RoPE + FP8 quantize + store to paged cache (C128 always head_dim=512)
  const void* norm_weight_ptr = norm_weight->GetPtr<void>();
  void* cache = layer_kv_cache_ptr->GetPtr<void*>()[1 + layer_idx_ * 2];
  const int32_t* compressed_out_loc_ptr = compressed_out_loc->GetPtr<int32_t>();
  DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype_, 512, kv_compressed_output, norm_weight_ptr, seq_lens_ptr,
                                    freqs_cis_, batch_size, ForwardMode::CompressDecode, rms_norm_eps_, compress_ratio_,
                                    cache, compressed_out_loc_ptr, page_stride_, compressed_page_bits_, stream);

  return Status();
}

#undef DISPATCH_FLASH_BY_COMPRESSOR_DTYPE
#undef DISPATCH_COMPRESS_STORE_BY_KVTYPE

}  // namespace ksana_llm
