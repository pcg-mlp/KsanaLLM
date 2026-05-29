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
using Wrapper = CompressorWrapper<float, __nv_bfloat16>;
using RopeWrapper = CompressorWrapper<__nv_bfloat16, __nv_bfloat16>;

// Dispatch CompressFusedNormRopeStore by kv_cache_dtype
#define DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype, head_dim, ...)                                        \
  switch (kv_cache_dtype) {                                                                                     \
    case TYPE_FP8_DS_MLA:                                                                                       \
      RopeWrapper::template CompressFusedNormRopeStore<head_dim, llm_kernels::utils::KVCacheType::kFp8DsMla>(   \
          __VA_ARGS__);                                                                                         \
      break;                                                                                                    \
    case TYPE_FP8_E4M3:                                                                                         \
      RopeWrapper::template CompressFusedNormRopeStore<head_dim, llm_kernels::utils::KVCacheType::kFp8E4M3>(    \
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

  DISPATCH_BY_2_DTYPE(inter_data_type_, InitYarnRotaryEmbedding, rotary_embedding_cuda_, rope_scaling_factor_config,
                      freqs_cis_, compress_rope_theta, rope_head_dim, max_position_embeddings, head_dim_, n_heads,
                      /*is_neox*/ false, context_->device->GetPrimaryComputeStream().Get());
}

// =============================================================================
// Forward
// =============================================================================
Status CompressorLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  const auto& kv_scale_input = input_tensors[1];
  KLLM_CHECK_WITH_INFO(kv_scale_input.dtype == DataType::TYPE_FP32, "kv_score_input must be float type");

  const bool is_prefill = (input_tensors.size() >= 10);
  if (is_prefill) {
    if (compress_ratio_ == 4) {
      return ForwardPrefillC4(input_tensors, output_tensors);
    } else {
      return ForwardPrefillC128(input_tensors, output_tensors);
    }
  } else {
    if (compress_ratio_ == 4) {
      return ForwardDecodeC4(input_tensors, output_tensors);
    } else {
      return ForwardDecodeC128(input_tensors, output_tensors);
    }
  }
}

// =============================================================================
// Prefill – C4 (compress_ratio = 4, overlap mode)
// =============================================================================
Status CompressorLayer::ForwardPrefillC4(const std::vector<Tensor>& input_tensors,
                                         std::vector<Tensor>& output_tensors) {
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  auto input_iter = input_tensors.cbegin();
  void* kv_score_buffer = (input_iter++)->GetPtr<void*>()[layer_idx_];
  const Tensor& kv_score_input_tensor = *input_iter++;
  const void* kv_score_input = kv_score_input_tensor.GetPtr<void>();
  const void* score_bias = (input_iter++)->GetPtr<void>();
  const Tensor& indices_tensor = *input_iter++;
  const int32_t* indices = indices_tensor.GetPtr<int32_t>();
  const void* extra = (input_iter++)->GetPtr<void>();
  const Tensor& compress_plan_tensor = *input_iter++;
  auto* compress_plan = compress_plan_tensor.GetPtr<PrefillPlan>();
  const Tensor& write_plan_tensor = *input_iter++;
  auto* write_plan = write_plan_tensor.GetPtr<PrefillPlan>();
  const int32_t* seq_lens = (input_iter++)->GetPtr<int32_t>();
  const int32_t* extend_lens = (input_iter++)->GetPtr<int32_t>();

  void* kv_compressed_output = output_tensors[0].GetPtr<void>();

  const uint32_t num_tokens = static_cast<uint32_t>(kv_score_input_tensor.shape[0]);

  // Step 1: build plan if needed
  if (compress_plan_tensor.shape[2] == 0) {
    const uint32_t batch_size = static_cast<uint32_t>(indices_tensor.shape[0]);
    Wrapper::PlanPrefillCompress(compress_plan, write_plan, seq_lens, extend_lens, batch_size, num_tokens,
                                 compress_ratio_, is_overlap_, stream);
  }

  // Step 2: prefill compression
  const uint32_t num_compress = static_cast<uint32_t>(compress_plan_tensor.shape[0]);
  const uint32_t num_write = static_cast<uint32_t>(write_plan_tensor.shape[0]);
  if (head_dim_ == 512) {
    Wrapper::template FlashC4Prefill<512>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                          compress_plan, write_plan, extra, num_compress, num_write, stream);
  } else {
    Wrapper::template FlashC4Prefill<128>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                          compress_plan, write_plan, extra, num_compress, num_write, stream);
  }

  // Step 3: Fused RMSNorm + RoPE + FP8 quantize + store to paged cache
  const void* norm_weight = (input_iter++)->GetPtr<void>();
  const Tensor& layer_kv_cache_ptr_tensor = *input_iter++;
  const int32_t* compressed_out_loc = (input_iter++)->GetPtr<int32_t>();
  void* cache = layer_kv_cache_ptr_tensor.GetPtr<void*>()[1 + layer_idx_ * 2];
  if (head_dim_ == 512) {
    DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype_, 512, kv_compressed_output, norm_weight, compress_plan,
                                      freqs_cis_, num_compress, ForwardMode::CompressExtend, rms_norm_eps_,
                                      compress_ratio_, cache, compressed_out_loc, page_stride_, compressed_page_bits_,
                                      stream);
  } else {
    RopeWrapper::template IndexerFusedNormRopeStore<128>(
        kv_compressed_output, norm_weight, compress_plan, freqs_cis_, num_compress, ForwardMode::CompressExtend,
        rms_norm_eps_, compress_ratio_, cache, compressed_out_loc, page_stride_, compressed_page_bits_, stream);
  }

  return Status();
}

// =============================================================================
// Prefill – C128 (compress_ratio = 128, non-overlap mode)
// =============================================================================
Status CompressorLayer::ForwardPrefillC128(const std::vector<Tensor>& input_tensors,
                                           std::vector<Tensor>& output_tensors) {
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  auto input_iter = input_tensors.cbegin();
  void* kv_score_buffer = (input_iter++)->GetPtr<void*>()[layer_idx_];
  const auto& kv_score_input_tensor = *input_iter++;
  const void* kv_score_input = kv_score_input_tensor.GetPtr<void>();
  const void* score_bias = (input_iter++)->GetPtr<void>();
  const auto& indices_tensor = *input_iter++;
  const int32_t* indices = indices_tensor.GetPtr<int32_t>();
  const int32_t* load_indices = (input_iter++)->GetPtr<int32_t>();
  const Tensor& compress_plan_tensor = *input_iter++;
  auto* compress_plan = compress_plan_tensor.GetPtr<PrefillPlan>();
  const Tensor& write_plan_tensor = *input_iter++;
  auto* write_plan = write_plan_tensor.GetPtr<PrefillPlan>();
  const int32_t* seq_lens = (input_iter++)->GetPtr<int32_t>();
  const int32_t* extend_lens = (input_iter++)->GetPtr<int32_t>();

  void* kv_compressed_output = output_tensors[0].GetPtr<void>();

  const uint32_t num_tokens = static_cast<uint32_t>(kv_score_input_tensor.shape[0]);

  // Step 1: build plan if needed
  if (compress_plan_tensor.shape[2] == 0) {
    const uint32_t batch_size = static_cast<uint32_t>(indices_tensor.shape[0]);
    Wrapper::PlanPrefillCompress(compress_plan, write_plan, seq_lens, extend_lens, batch_size, num_tokens,
                                 compress_ratio_, is_overlap_, stream);
  }

  // Step 2: prefill compression (C128 always has head_dim=512)
  const uint32_t num_compress = static_cast<uint32_t>(compress_plan_tensor.shape[0]);
  const uint32_t num_write = static_cast<uint32_t>(write_plan_tensor.shape[0]);
  Wrapper::template FlashC128Prefill<512>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                          load_indices, compress_plan, write_plan, num_compress, num_write, stream);

  // Step 3: Fused RMSNorm + RoPE + FP8 quantize + store to paged cache (C128 always head_dim=512)
  const void* norm_weight = (input_iter++)->GetPtr<void>();
  const Tensor& layer_kv_cache_ptr_tensor = *input_iter++;
  const int32_t* compressed_out_loc = (input_iter++)->GetPtr<int32_t>();
  void* cache = layer_kv_cache_ptr_tensor.GetPtr<void*>()[1 + layer_idx_ * 2];
  DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype_, 512, kv_compressed_output, norm_weight, compress_plan, freqs_cis_,
                                    num_compress, ForwardMode::CompressExtend, rms_norm_eps_, compress_ratio_, cache,
                                    compressed_out_loc, page_stride_, compressed_page_bits_, stream);

  return Status();
}

// =============================================================================
// Decode – C4
// =============================================================================
Status CompressorLayer::ForwardDecodeC4(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  auto input_iter = input_tensors.cbegin();
  void* kv_score_buffer = (input_iter++)->GetPtr<void*>()[layer_idx_];
  const void* kv_score_input = (input_iter++)->GetPtr<void>();
  const void* score_bias = (input_iter++)->GetPtr<void>();
  const auto& indices_tensor = *input_iter++;
  const int32_t* indices = indices_tensor.GetPtr<int32_t>();
  const int32_t* seq_lens = (input_iter++)->GetPtr<int32_t>();
  const int32_t* extra = (input_iter++)->GetPtr<int32_t>();

  void* kv_compressed_output = output_tensors[0].GetPtr<void>();

  // Step 1: decode compression
  const uint32_t batch_size = static_cast<uint32_t>(indices_tensor.shape[0]);
  if (head_dim_ == 512) {
    Wrapper::template FlashC4Decode<512>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                         seq_lens, extra, batch_size, stream);
  } else {
    Wrapper::template FlashC4Decode<128>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                         seq_lens, extra, batch_size, stream);
  }

  // Step 2: Fused RMSNorm + RoPE + FP8 quantize + store to paged cache
  const void* norm_weight = (input_iter++)->GetPtr<void>();
  const Tensor& layer_kv_cache_ptr_tensor = *input_iter++;
  const int32_t* compressed_out_loc = (input_iter++)->GetPtr<int32_t>();
  void* cache = layer_kv_cache_ptr_tensor.GetPtr<void*>()[1 + layer_idx_ * 2];
  if (head_dim_ == 512) {
    DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype_, 512, kv_compressed_output, norm_weight, seq_lens, freqs_cis_,
                                      batch_size, ForwardMode::CompressDecode, rms_norm_eps_, compress_ratio_, cache,
                                      compressed_out_loc, page_stride_, compressed_page_bits_, stream);
  } else {
    RopeWrapper::template IndexerFusedNormRopeStore<128>(
        kv_compressed_output, norm_weight, seq_lens, freqs_cis_, batch_size, ForwardMode::CompressDecode, rms_norm_eps_,
        compress_ratio_, cache, compressed_out_loc, page_stride_, compressed_page_bits_, stream);
  }

  return Status();
}

// =============================================================================
// Decode – C128
// =============================================================================
Status CompressorLayer::ForwardDecodeC128(const std::vector<Tensor>& input_tensors,
                                          std::vector<Tensor>& output_tensors) {
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  auto input_iter = input_tensors.cbegin();
  void* kv_score_buffer = (input_iter++)->GetPtr<void*>()[layer_idx_];
  const void* kv_score_input = (input_iter++)->GetPtr<void>();
  const void* score_bias = (input_iter++)->GetPtr<void>();
  const auto& indices_tensor = *input_iter++;
  const int32_t* indices = indices_tensor.GetPtr<int32_t>();
  const int32_t* seq_lens = (input_iter++)->GetPtr<int32_t>();

  void* kv_compressed_output = output_tensors[0].GetPtr<void>();

  // Step 1: decode compression (C128 always has head_dim=512)
  const uint32_t batch_size = static_cast<uint32_t>(indices_tensor.shape[0]);
  Wrapper::template FlashC128Decode<512>(kv_score_buffer, kv_score_input, kv_compressed_output, score_bias, indices,
                                         seq_lens, batch_size, stream);

  // Step 2: Fused RMSNorm + RoPE + FP8 quantize + store to paged cache (C128 always head_dim=512)
  const void* norm_weight = (input_iter++)->GetPtr<void>();
  const Tensor& layer_kv_cache_ptr_tensor = *input_iter++;
  const int32_t* compressed_out_loc = (input_iter++)->GetPtr<int32_t>();
  void* cache = layer_kv_cache_ptr_tensor.GetPtr<void*>()[1 + layer_idx_ * 2];
  DISPATCH_COMPRESS_STORE_BY_KVTYPE(kv_cache_dtype_, 512, kv_compressed_output, norm_weight, seq_lens, freqs_cis_,
                                    batch_size, ForwardMode::CompressDecode, rms_norm_eps_, compress_ratio_, cache,
                                    compressed_out_loc, page_stride_, compressed_page_bits_, stream);

  return Status();
}

}  // namespace ksana_llm
