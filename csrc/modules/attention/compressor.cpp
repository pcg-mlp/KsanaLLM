/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/attention/compressor.h"

#include <fmt/core.h>

#include "layers/cast_layer.h"
#include "layers/compressor_layer.h"
#include "layers/nvidia/dsv4_tail_copy.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta.h"
#include "profiler/profile_event.h"
#include "utils/logger.h"

namespace ksana_llm {

namespace {
// Compute the local layer index for a given global layer index, taking pipeline
// parallelism (lower/upper_layer_idx) and MTP layers into account. This mirrors
// dsv4_mqa.cpp's attn_config.idx assignment so that per-layer tensors (which are
// populated by local index) can be indexed correctly inside Compressor.
int ComputeLocalLayerIdx(int global_layer_idx, const PipelineConfig& pipeline_config, int num_layer) {
  int local_layer_idx = global_layer_idx - pipeline_config.lower_layer_idx;
  if (global_layer_idx >= num_layer) {
    // MTP / next-N layer: place after the main layers on this node.
    local_layer_idx =
        pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + global_layer_idx - num_layer + 1;
  }
  return local_layer_idx;
}
}  // namespace

Compressor::Compressor(int layer_idx, bool is_in_indexer, const LayerCreationContext& creation_context,
                       const ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx),
      local_layer_idx_(
          ComputeLocalLayerIdx(layer_idx, creation_context.pipeline_config,
                               static_cast<int>(model_creation_config.attn_config.model_config.num_layer))),
      is_in_indexer_(is_in_indexer) {
  const auto& attn_config = model_creation_config.attn_config;
  const auto& model_config = attn_config.model_config;
  const auto& runtime_config = creation_context.runtime_config;

  compressor_dtype_ = runtime_config.compressor_config.state_dtype;
  KLLM_CHECK_WITH_INFO(compressor_dtype_ == TYPE_FP32 || compressor_dtype_ == TYPE_BF16,
                       "Compressor requires state_dtype to be either FP32 or BF16");

  KLLM_CHECK_WITH_INFO(
      layer_idx >= 0 && layer_idx < static_cast<int>(model_config.dsv4_attn_config.compress_ratios.size()),
      "layer_idx out of bounds for compress_ratios array");
  compress_ratio_ = model_config.dsv4_attn_config.compress_ratios[layer_idx];
  coff_ = 1 + (compress_ratio_ == 4);
  KLLM_CHECK_WITH_INFO(compress_ratio_ > 0, "Compressor requires compress_ratio > 0");
  KLLM_CHECK_WITH_INFO(!is_in_indexer || compress_ratio_ == 4, "Indexer requires compress_ratio == 4");
  const int n_heads = is_in_indexer ? model_config.dsa_config.index_n_heads
                                    : attn_config.head_num_per_tp;  // Required to be passed but not actually used
  head_dim_ = is_in_indexer ? model_config.dsa_config.index_head_dim : attn_config.size_per_head;
  per_token_bytes_ = static_cast<int>(2 * coff_ * head_dim_ * GetTypeSize(compressor_dtype_));

  const std::string layer_prefix = is_in_indexer
                                       ? fmt::format("model.layers.{}.self_attn.indexer.compressor", layer_idx)
                                       : fmt::format("model.layers.{}.self_attn.compressor", layer_idx);
  const auto& linear_compute_backend = model_config.quant_config.backend;
  const float layernorm_eps = model_creation_config.layernorm_config.layernorm_eps;

  // ── Weights ──────────────────────────────────────────────────────────────────
  // wkv_gate: [hidden_size → 2 * coff * head_dim]
  wkv_gate_proj_ =
      std::make_unique<Linear>(layer_prefix + ".wkv_gate.weight", creation_context, linear_compute_backend);

  // ape: [ratio, coff * head_dim] — kernel always reads as FP32
  ape_weight_ = creation_context.base_weight->GetModelWeights(layer_prefix + ".ape");
  KLLM_CHECK_WITH_INFO(ape_weight_.dtype == TYPE_FP32, "APE weight must be FP32");

  // norm weight tensor (passed directly to CompressorLayer::Forward as input)
  norm_weight_ = creation_context.base_weight->GetModelWeights(layer_prefix + ".norm.weight");

  // ── CastLayer initialisation (bf16 → fp32 for kv_score_input) ──────────
  // Only create the cast layer when the compressor uses FP32 internally.
  if (compressor_dtype_ == TYPE_FP32) {
    cast_layer_ = std::make_unique<CastLayer>();
    cast_layer_->Init({}, runtime_config, creation_context.context, creation_context.rank);
  }

  // ── CompressorLayer initialisation ───────────────────────────────────────────
  compressor_layer_ = std::make_unique<CompressorLayer>();

  std::vector<std::any> params;
  params.push_back(compress_ratio_);                                                       // compress_ratio
  params.push_back(n_heads);                                                               // n_heads
  params.push_back(static_cast<int>(head_dim_));                                           // head_dim
  params.push_back(static_cast<int>(model_config.dsv4_attn_config.qk_rope_head_dim));      // rope_head_dim
  params.push_back(layernorm_eps);                                                         // rms_norm_eps
  params.push_back(attn_config.idx);                                                       // local_layer_idx
  params.push_back(attn_config.position_encoding);                                         // position_encoding
  params.push_back(attn_config.compress_rope_theta);                                       // compress_rope_theta
  params.push_back(attn_config.compress_cos_sin_cache_ptr);                                // freqs_cis
  params.push_back(model_config.rope_scaling_factor_config);                               // rope_scaling
  params.push_back(attn_config.max_position_embeddings);                                   // max_position_embeddings
  params.push_back(static_cast<int>(runtime_config.attn_backend_config.block_token_num));  // block_token_num
  params.push_back(static_cast<int64_t>(runtime_config.attn_backend_config.block_size));   // page_stride
  params.push_back(attn_config.kv_cache_dtype);                                            // kv_cache_dtype (extra)
  params.push_back(compressor_dtype_);                                                     // compressor_dtype

  compressor_layer_->Init(params, runtime_config, creation_context.context, creation_context.rank);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward: linear projection + compress
// ─────────────────────────────────────────────────────────────────────────────

Status Compressor::Forward(const Tensor& x, const Tensor& workspace, Tensor& kv_compressed_output,
                           ForwardingContext& forwarding_context) {
  const int rank = forwarding_context.GetCurrentRank();
  PROFILE_EVENT_SCOPE(compressor_forward, "compressor_forward", rank);

  // x.shape = [total_tokens, hidden_dim]
  const size_t total_tokens = x.shape[0];
  // Used to locate unused workspace buffers
  // Assume the given workspace is sufficient since n_heads and head_dim in compressor is small
  size_t workspace_offset = 0;

  // Step 1: kv_score = wkv_gate_proj_(x)
  // Input: x [T, hidden_dim], Output: kv_score [T, 2 * coff * head_dim]
  Tensor kv_score_input = workspace.GetView({total_tokens, 2 * coff_ * head_dim_}, workspace_offset);
  workspace_offset += kv_score_input.GetElementNumber();
  std::vector<Tensor> wkv_gate_proj_outputs = {kv_score_input};
  STATUS_CHECK_RETURN(wkv_gate_proj_->Forward(x, wkv_gate_proj_outputs));

  // Step 2: Convert kv_score from bf16 to fp32 (only when compressor uses FP32 internally)
  if (compressor_dtype_ == TYPE_FP32) {
    KLLM_CHECK(workspace.dtype == wkv_gate_proj_outputs[0].dtype);
    Tensor fp32_cast_output =
        workspace.GetView({kv_score_input.GetElementNumber() * 2 /*for bf16 to fp32*/}, workspace_offset);
    STATUS_CHECK_RETURN(cast_layer_->Forward(&wkv_gate_proj_outputs[0], 0, 1, 1, &fp32_cast_output));
    kv_score_input = fp32_cast_output;
  }

  // Step 3: kv_compressed_output = compress_layer_(kv_score_input)
  // Input: kv_score_input [T, 2 * coff * head_dim] (FP32 or BF16), Output: kv_compressed_output [T, head_dim]
  const auto& model_input = forwarding_context.GetModelInput();

  // Retrieve DSV4-specific shared and per-batch compressor data from the attention info hierarchy.
  auto& dsv4_shared = static_cast<DeepSeekV4SharedAttentionMeta&>(*model_input->shared_attention_meta);
  auto& dsv4_flash_meta = static_cast<DeepSeekV4FlashAttentionMeta&>(*model_input->flash_meta);
  CompressorSharedAttentionMeta& compressor_shared = dsv4_shared.compressor;
  CompressorAttentionMeta& flash_compressor = dsv4_flash_meta.compressor;

  // Find the single-token decode page_meta (q_seq_len == 1) for the paged compressor path.
  // In MTP scenarios there may be higher-q_seq_len buckets before it.
  DeepSeekV4PagedAttentionMeta* dsv4_paged_ptr = nullptr;
  for (auto& pi : model_input->page_metas) {
    if (pi->q_seq_len == 1) {
      dsv4_paged_ptr = dynamic_cast<DeepSeekV4PagedAttentionMeta*>(pi.get());
      break;
    }
  }

  // Use the compressor's own token counts rather than model_input->dp_context/decode_tokens.
  // flash_tokens = prefill + MTP decode (all !IsPaged reqs); paged_tokens = std decode (IsPaged reqs).
  // model_input->dp_context_tokens only counts kFlash reqs (prefill), missing MTP decode.
  const size_t flash_tokens = flash_compressor.dp_tokens;
  const size_t paged_tokens = dsv4_paged_ptr ? dsv4_paged_ptr->compressor.dp_tokens : 0;

  // Prefix cache tail read-copy: 把 prefix 末 block 的 tail kv_score 复制到 state buffer。
  if (compress_ratio_ == 4) {
    if (const int num_req = static_cast<int>(compressor_shared.tail_read_block_tensor.GetElementNumber());
        num_req > 0) {
      auto& mqa_shared = *model_input->shared_attention_meta->GetMqa();
      Tensor* tail_layer_base_tensor = is_in_indexer_ ? &mqa_shared.layer_tail_indexer_cache_ptr_tensor
                                                      : &mqa_shared.layer_tail_attn_cache_ptr_tensor;
      void* const tail_layer_base = tail_layer_base_tensor->GetPtr<void*>()[1 + local_layer_idx_];
      Tensor& state_layer_ptr_tensor = is_in_indexer_ ? compressor_shared.layer_c4_indexer_state_ptr_tensor
                                                      : compressor_shared.layer_c4_state_ptr_tensor;
      void* const state_layer_base = state_layer_ptr_tensor.GetPtr<void*>()[local_layer_idx_];
      const int64_t block_cache_size = static_cast<int64_t>(mqa_shared.swa_block_size);
      constexpr int kTailTokens = 4;
      KLLM_LOG_DEBUG << fmt::format(
          "tail read-copy: layer={} is_in_indexer={} num_req={} per_token_bytes={} block_cache_size={}", layer_idx_,
          is_in_indexer_, num_req, per_token_bytes_, block_cache_size);
      CopyTailToState(state_layer_base, tail_layer_base, compressor_shared.tail_read_block_tensor.GetPtr<int32_t>(),
                      compressor_shared.tail_target_indices_tensor.GetPtr<int32_t>(), num_req, block_cache_size,
                      per_token_bytes_, kTailTokens,
                      forwarding_context.GetContext()->device->GetPrimaryComputeStream().Get());
    }
  }

  if (flash_tokens > 0) {
    Tensor context_kv_score_input = kv_score_input.GetView({flash_tokens, kv_score_input.shape[1]});
    Tensor context_output = kv_compressed_output.GetView({flash_tokens, head_dim_});
    if (compress_ratio_ == 4) {
      // ForwardPrefillC4
      const bool plan_initialized = (compressor_shared.c4_compress_plan_tensor.shape[2] != 0);
      Tensor* kv_score_buf_tensor = is_in_indexer_ ? &compressor_shared.layer_c4_indexer_state_ptr_tensor
                                                   : &compressor_shared.layer_c4_state_ptr_tensor;
      Tensor* kv_cache_ptr_tensor = is_in_indexer_
                                        ? &model_input->shared_attention_meta->GetDSA()->layer_indexer_kv_cache_ptr
                                        : &model_input->shared_attention_meta->GetStandard()->layer_kv_cache_ptr;
      STATUS_CHECK_RETURN(compressor_layer_->ForwardPrefillC4(
          kv_score_buf_tensor, &context_kv_score_input, &ape_weight_, &flash_compressor.c4_indices_tensor,
          &flash_compressor.c4_indices_bundle_tensor, &compressor_shared.c4_compress_plan_tensor,
          &compressor_shared.c4_write_plan_tensor, &flash_compressor.seq_lens_tensor,
          &flash_compressor.extend_lens_tensor, &norm_weight_, kv_cache_ptr_tensor, &flash_compressor.c4_out_loc_tensor,
          &context_output, plan_initialized));
      compressor_shared.c4_compress_plan_tensor.shape[2] = 1;
      compressor_shared.c4_write_plan_tensor.shape[2] = 1;
    } else {
      // ForwardPrefillC128 (is_in_indexer_ never true for C128)
      const bool plan_initialized = (compressor_shared.c128_compress_plan_tensor.shape[2] != 0);
      STATUS_CHECK_RETURN(compressor_layer_->ForwardPrefillC128(
          &compressor_shared.layer_c128_state_ptr_tensor, &context_kv_score_input, &ape_weight_,
          &flash_compressor.c128_indices_tensor, &flash_compressor.c128_load_indices_tensor,
          &compressor_shared.c128_compress_plan_tensor, &compressor_shared.c128_write_plan_tensor,
          &flash_compressor.seq_lens_tensor, &flash_compressor.extend_lens_tensor, &norm_weight_,
          &model_input->shared_attention_meta->GetStandard()->layer_kv_cache_ptr, &flash_compressor.c128_out_loc_tensor,
          &context_output, plan_initialized));
      compressor_shared.c128_compress_plan_tensor.shape[2] = 1;
      compressor_shared.c128_write_plan_tensor.shape[2] = 1;
    }
  }

  if (paged_tokens > 0 && dsv4_paged_ptr != nullptr) {
    CompressorAttentionMeta& paged_compressor = dsv4_paged_ptr->compressor;
    Tensor decode_kv_score_input =
        kv_score_input.GetView({paged_tokens, kv_score_input.shape[1]}, flash_tokens * kv_score_input.shape[1]);
    Tensor decode_output = kv_compressed_output.GetView({paged_tokens, head_dim_}, flash_tokens * head_dim_);
    if (compress_ratio_ == 4) {
      // ForwardDecodeC4
      Tensor* kv_score_buf_tensor = is_in_indexer_ ? &compressor_shared.layer_c4_indexer_state_ptr_tensor
                                                   : &compressor_shared.layer_c4_state_ptr_tensor;
      Tensor* kv_cache_ptr_tensor = is_in_indexer_
                                        ? &model_input->shared_attention_meta->GetDSA()->layer_indexer_kv_cache_ptr
                                        : &model_input->shared_attention_meta->GetStandard()->layer_kv_cache_ptr;
      STATUS_CHECK_RETURN(compressor_layer_->ForwardDecodeC4(
          kv_score_buf_tensor, &decode_kv_score_input, &ape_weight_, &paged_compressor.c4_indices_tensor,
          &paged_compressor.seq_lens_tensor, &paged_compressor.c4_indices_prev_tensor, &norm_weight_,
          kv_cache_ptr_tensor, &paged_compressor.c4_out_loc_tensor, &decode_output));
    } else {
      // ForwardDecodeC128 (is_in_indexer_ never true for C128)
      STATUS_CHECK_RETURN(compressor_layer_->ForwardDecodeC128(
          &compressor_shared.layer_c128_state_ptr_tensor, &decode_kv_score_input, &ape_weight_,
          &paged_compressor.c128_indices_tensor, &paged_compressor.seq_lens_tensor, &norm_weight_,
          &model_input->shared_attention_meta->GetStandard()->layer_kv_cache_ptr, &paged_compressor.c128_out_loc_tensor,
          &decode_output));
    }
  }

  // Prefix cache write-copy: 把 c4 层当前 step 的 kv_score 持久化到 tail 区，供下个请求复用。
  if (compress_ratio_ == 4) {
    if (const size_t num_tokens = compressor_shared.tail_write_indices_tensor.GetElementNumber(); num_tokens > 0) {
      KLLM_CHECK(num_tokens == total_tokens);
      auto& mqa_shared = *model_input->shared_attention_meta->GetMqa();
      Tensor* tail_layer_base_tensor = is_in_indexer_ ? &mqa_shared.layer_tail_indexer_cache_ptr_tensor
                                                      : &mqa_shared.layer_tail_attn_cache_ptr_tensor;
      // layer_tail_*_cache_ptr_tensor layout: [block_num(int64) | layer_0_base | layer_1_base | ...]
      // Slots 1..N correspond to local layer 0..N-1, so local_layer_idx_ must be used here.
      void* const tail_layer_base = tail_layer_base_tensor->GetPtr<void*>()[1 + local_layer_idx_];
      // Per-token bytes match a single row of kv_score_input.
      // tail 写入的目标位于 SWA pool, stride 使用 SWA pool 的 block_size。
      const int64_t block_cache_size = static_cast<int64_t>(mqa_shared.swa_block_size);
      KLLM_LOG_DEBUG << fmt::format(
          "tail write-copy: layer={} is_in_indexer={} num_tokens={} per_token_bytes={} block_cache_size={}", layer_idx_,
          is_in_indexer_, num_tokens, per_token_bytes_, block_cache_size);
      CopyKvScoreToTail(tail_layer_base,
                        kv_score_input.GetPtr<void>(),  // FP32 [total_tokens, 2*coff*head_dim]
                        compressor_shared.tail_write_indices_tensor.GetPtr<int32_t>(), static_cast<int>(num_tokens),
                        block_cache_size, per_token_bytes_,
                        forwarding_context.GetContext()->device->GetPrimaryComputeStream().Get());
    }
  }

  // Correctly set the output shape
  kv_compressed_output.shape = {total_tokens, head_dim_};  // Uncompressed tokens remain in place

  KLLM_LOG_DEBUG << fmt::format("Compressor Forward completed: layer={}", layer_idx_);

  return Status();
}

}  // namespace ksana_llm
