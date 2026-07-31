/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/kimi_k3_model.h"

#include "device/device_types.h"
#include "device/device_utils.h"
#include "profiler/profile_event.h"
#include "runtime/profile_metrics.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

KimiK3DecoderLayer::KimiK3DecoderLayer(int layer_idx, bool is_moe, LayerCreationContext& creation_context,
                                       ModelCreationConfig& model_creation_config, MlaBuffers& mla_buffers,
                                       TensorBuffer* moe_buffer)
    : is_moe_(is_moe),
      is_block_write_layer_(
          layer_idx %
              static_cast<int>(model_creation_config.attn_config.model_config.attn_res_config.attn_res_block_size) ==
          0),
      layer_idx_(layer_idx),
      rank_(creation_context.rank),
      mla_buffers_(mla_buffers),
      moe_buffer_(moe_buffer) {
  MoeScaleNormMode moe_scale_norm_mode = MoeScaleNormMode::RE_NORM;

  const std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  attn_res_ = std::make_unique<AttnRes>(layer_prefix + ".self_attention_res", creation_context,
                                        layer_prefix + ".input_layernorm.weight");
  mlp_res_ = std::make_unique<AttnRes>(layer_prefix + ".mlp_res", creation_context,
                                       layer_prefix + ".post_attention_layernorm.weight");

  add_ = std::make_unique<Add>(creation_context);

  if (is_moe) {
    shared_mlp_ = std::make_unique<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config,
                                                  ".mlp.shared_expert.{}.weight");
    expert_gate_ = std::make_unique<Linear>(layer_prefix + ".mlp.gate.weight", creation_context,
                                            model_creation_config.attn_config.model_config.quant_config.backend);
    if (model_creation_config.attn_config.model_config.moe_config.use_e_score_correction_bias) {
      moe_ = std::make_unique<MoE>(
          layer_idx, layer_prefix + ".mlp.experts.up_gate_proj.weight", layer_prefix + ".mlp.experts.down_proj.weight",
          layer_prefix + ".mlp.gate.e_score_correction_bias", creation_context, moe_scale_norm_mode);
    } else {
      moe_ =
          std::make_unique<MoE>(layer_idx, layer_prefix + ".mlp.experts.up_gate_proj.weight",
                                layer_prefix + ".mlp.experts.down_proj.weight", creation_context, moe_scale_norm_mode);
    }
  } else {
    mlp_ = std::make_unique<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config);
  }

  // MLA must be initialized after Linear layers so it can reuse their workspace.
  mla_ = std::make_unique<MultiHeadLatentAttention>(layer_idx, /*is_neox*/ false, creation_context,
                                                    model_creation_config, mla_buffers_);

  tp_comm_ = std::make_shared<TpCommunicator>();
}

Status KimiK3DecoderLayer::Forward(std::vector<Tensor>& prefix_buffer, std::vector<Tensor>& residual_buffer,
                                   const bool is_multi_token_forward, ForwardingContext& forwarding_context,
                                   bool is_first_layer) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);

  // ==================== ATTN block start ====================
  // ATTN residual + fused RMSNorm
  std::vector<Tensor> attn_res_tensors{prefix_buffer[0], residual_buffer[0]};
  if (!is_first_layer) {
    attn_res_tensors.push_back(hidden_buffer_tensors_0[0]);
  }
  STATUS_CHECK_RETURN(
      attn_res_->Forward(attn_res_tensors, hidden_buffer_tensors_0, /*write_block=*/is_block_write_layer_));
  if (is_block_write_layer_) {
    ++residual_buffer[0].shape[1];
  }

  // ATTN core
  MODEL_PERF_OP_BEGIN(forwarding_context.current_forward_metrics, attn_t);
  mla_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, tp_comm_, is_multi_token_forward, forwarding_context);
  STATUS_CHECK_RETURN(
      tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context));
  MODEL_PERF_OP_END(forwarding_context.current_forward_metrics, ForwardProfileOp::kAttn, attn_t);
  // ==================== ATTN block end ======================

  // ==================== FFN block start =====================
  // FFN residual + fused RMSNorm
  std::vector<Tensor> mlp_res_tensors{prefix_buffer[0], residual_buffer[0]};
  if (is_block_write_layer_) {
    MemcpyAsync(prefix_buffer[0].GetPtr<void>(), hidden_buffer_tensors_0[0].GetPtr<void>(),
                hidden_buffer_tensors_0[0].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
                forwarding_context.GetContext()->device->GetPrimaryComputeStream());
  } else {
    mlp_res_tensors.push_back(hidden_buffer_tensors_0[0]);
  }
  STATUS_CHECK_RETURN(mlp_res_->Forward(mlp_res_tensors, hidden_buffer_tensors_0, /*write_block=*/false));

  // FFN core
  MODEL_PERF_OP_BEGIN(forwarding_context.current_forward_metrics, ffn_t);
  STATUS_CHECK_RETURN(
      CommonMlp(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  STATUS_CHECK_RETURN(
      tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context));
  MODEL_PERF_OP_END(forwarding_context.current_forward_metrics, ForwardProfileOp::kFfn, ffn_t);
  // ==================== FFN block end ======================
  return Status();
}

Status KimiK3DecoderLayer::CommonMlp(std::vector<Tensor>& hidden_buffer_tensors_0,
                                     std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                     ForwardingContext& forwarding_context) {
  const size_t seq_len = hidden_buffer_tensors_0[0].shape[0];
  const size_t hidden_units = hidden_buffer_tensors_0[0].shape[1];

  PROFILE_EVENT_SCOPE(K3_CommonMlp_seq_len_,
                      fmt::format("K3_CommonMlp_seq_len_{}_hidden_units_{}", seq_len, hidden_units),
                      forwarding_context.GetCurrentRank());

  if (!is_moe_) {
    PROFILE_EVENT_SCOPE(CommonMlp, "CommonMlp", forwarding_context.GetCurrentRank());
    mlp_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context);
    return Status();
  }

  CREATE_BUFFER_SCOPE(moe_buffer_tensors, moe_buffer_);
  auto& gated_buffer_ = moe_buffer_tensors;

  {
    PROFILE_EVENT_SCOPE(expert_gate, "expert_gate", forwarding_context.GetCurrentRank());
    STATUS_CHECK_RETURN(expert_gate_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0]));
  }

  {
    PROFILE_EVENT_SCOPE(moe, "MOE", forwarding_context.GetCurrentRank());
    STATUS_CHECK_RETURN(
        moe_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], reduce_buffer_tensors[0], moe_buffer_tensors));
  }

  {
    PROFILE_EVENT_SCOPE(CommonShareMlp, "CommonShareMlp", forwarding_context.GetCurrentRank());
    STATUS_CHECK_RETURN(shared_mlp_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward,
                                             forwarding_context));
  }

  {
    PROFILE_EVENT_SCOPE(add_layer, "add_layer", forwarding_context.GetCurrentRank());
    STATUS_CHECK_RETURN(add_->Forward(reduce_buffer_tensors[0], moe_buffer_tensors[0], reduce_buffer_tensors));
  }
  return Status();
}

/**********************************************************
 * KimiK3Model
 ***********************************************************/

KimiK3Model::KimiK3Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                         std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                         std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
    : CommonModel(model_config, runtime_config, rank, context),
      first_k_dense_replace_(model_config.moe_config.first_k_dense_replace) {
  ModelRunConfig model_run_config;
  model_run_config.position_encoding =
      model_config.mla_config.mla_use_nope ? PositionEncoding::NO_ROPE : PositionEncoding::ROPE;

  CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);
}

Status KimiK3Model::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  MultiHeadLatentAttention::CreateBuffers(CommonModel::GetBufferManager(), model_creation_config.attn_config,
                                          creation_context.runtime_config, mla_buffers_);
  const DataType weight_type = model_creation_config.attn_config.model_config.weight_data_type;
  const size_t max_token_num = creation_context.runtime_config.max_step_token_num;
  const size_t hidden_units = model_creation_config.attn_config.model_config.hidden_units;
  const size_t moe_buffer_size = max_token_num * hidden_units;
  moe_buffer_ = CommonModel::GetBufferManager()->CreateBufferTensor("moe_buffer_", {moe_buffer_size}, weight_type);

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; ++layer_idx) {
    const bool is_moe = layer_idx >= first_k_dense_replace_;
    layers_[layer_idx] = std::make_shared<KimiK3DecoderLayer>(layer_idx, is_moe, creation_context,
                                                              model_creation_config, mla_buffers_, moe_buffer_);
  }
  if (context_->global->IsSamplingNode()) {
    output_attn_res_ = std::make_unique<AttnRes>("model.output_attn_res", creation_context);
  }
  return Status();
}

Status KimiK3Model::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  PROFILE_EVENT_SCOPE(K3_LayerForward_, "K3_LayerForward", forwarding_context.GetCurrentRank());
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;

  const auto& global_ctx = forwarding_context.GetContext()->global;
  const bool is_first_pp = global_ctx->IsStandalone() || global_ctx->IsDistributedMaster();

  const auto& pipeline_config = forwarding_context.GetPipelineConfig();
  const int lower_layer_idx = pipeline_config.lower_layer_idx;
  const int upper_layer_idx = pipeline_config.upper_layer_idx;
  const uint32_t block_size = model_config_.attn_res_config.attn_res_block_size;
  const uint32_t cur_valid_blocks = DivRoundUp<uint32_t>(upper_layer_idx + 1, block_size);
  const uint32_t prev_valid_blocks = is_first_pp ? 0 : DivRoundUp<uint32_t>(lower_layer_idx, block_size);

  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
  const size_t num_tokens = residual_buffer[0].shape[0];
  const size_t hidden_units = model_config_.hidden_units;

  CREATE_BUFFER_SCOPE(prefix_buffer, forwarding_context.GetForwardingBuffers()->prefix_residual_buffer);
  prefix_buffer[0].shape = {num_tokens, hidden_units};

  Stream& stream = forwarding_context.GetContext()->device->GetPrimaryComputeStream();

  if (is_first_pp) {
    // First PP: incoming residual_buffer holds [num_tokens, H] embeddings. Seed prefix, then
    // reinterpret residual_buffer as the empty bank [num_tokens, max_blocks, H].
    KLLM_CHECK(residual_buffer[0].shape.size() == 2 && residual_buffer[0].shape[1] == hidden_units);
    const size_t slot_bytes = hidden_units * GetTypeSize(residual_buffer[0].dtype);
    MemcpyAsync(prefix_buffer[0].GetPtr<void>(), residual_buffer[0].GetPtr<void>(), num_tokens * slot_bytes,
                MEMCPY_DEVICE_TO_DEVICE, stream);
  } else {
    KLLM_CHECK(residual_buffer[0].shape.size() == 2 && residual_buffer[0].shape[1] == prev_valid_blocks * hidden_units);
  }

  residual_buffer[0].shape = {num_tokens, prev_valid_blocks, hidden_units};

  if (run_mode == RunMode::kMain) {
    for (int layer_idx = lower_layer_idx; layer_idx <= upper_layer_idx; ++layer_idx) {
      STATUS_CHECK_RETURN(layers_[layer_idx]->Forward(prefix_buffer, residual_buffer, is_multi_token_forward,
                                                      forwarding_context,
                                                      /*is_first_layer=*/layer_idx == lower_layer_idx));
    }
    KLLM_CHECK(residual_buffer[0].shape[1] == cur_valid_blocks);
  }

  if (output_attn_res_) {
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
    std::vector<Tensor> output_attn_res_tensors{prefix_buffer[0], residual_buffer[0], hidden_buffer_tensors_0[0]};
    STATUS_CHECK_RETURN(
        output_attn_res_->Forward(output_attn_res_tensors, hidden_buffer_tensors_0, /*write_block=*/false));
    residual_buffer[0].shape = hidden_buffer_tensors_0[0].shape;
    MemcpyAsync(residual_buffer[0].GetPtr<void>(), hidden_buffer_tensors_0[0].GetPtr<void>(),
                hidden_buffer_tensors_0[0].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, stream);
  } else {
    // Flatten bank back to [num_tokens, max_blocks * H] for downstream/PP handoff.
    residual_buffer[0].shape = {num_tokens, cur_valid_blocks * hidden_units};
  }
  return Status();
}

}  // namespace ksana_llm
