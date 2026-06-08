/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/ffn/shared_expert_layer.h"

#include "models/base/buffer_manager.h"

namespace ksana_llm {

SharedExpertLayer::SharedExpertLayer(int layer_idx, LayerCreationContext& creation_context,
                                     ModelCreationConfig& model_creation_config,
                                     const std::string& weight_name_format) {
  // Determine configuration (same logic as TwoLayeredFFN::InitConfig)
  const std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
  const std::string up_gate_proj_weights_name =
      fmt::format("model.layers.{}" + weight_name_format, layer_idx, "gate_up_proj");

  enable_full_shared_expert_ = creation_context.runtime_config.enable_full_shared_expert;

  if (creation_context.base_weight->GetModelWeights(up_gate_proj_weights_name).GetElementNumber() > 0) {
    fuse_gate_up_proj_ = true;
    // Check if silu_mul can be fused into fp8 group quant before down projection
    const std::string down_proj_weights_name =
        fmt::format("model.layers.{}" + weight_name_format, layer_idx, "down_proj");
    fuse_silu_mul_ = creation_context.base_weight->GetModelWeights(down_proj_weights_name).dtype == TYPE_FP8_E4M3 &&
                     creation_context.model_config.quant_config.method == QUANT_BLOCK_FP8_E4M3;
  } else {
    fuse_gate_up_proj_ = false;
  }
  KLLM_LOG_INFO << fmt::format("fuse_silu_mul_: {}, fuse_gate_up_proj_: {}", fuse_silu_mul_, fuse_gate_up_proj_);

  // Use secondary_workspace_mgr for SharedExpert's Linear layers.
  // This ensures their GEMM workspace is independent from the primary stream layers (MoE, Attention),
  // allowing safe parallel execution on SecondaryComputeStream.
  LayerCreationContext ctx = creation_context;
  if (creation_context.secondary_workspace_mgr) {
    ctx.workspace_mgr = creation_context.secondary_workspace_mgr;
  }

  // Create Linear layers (same logic as TwoLayeredFFN::InitLayers)
  LinearComputeBackend linear_compute_backend = model_creation_config.attn_config.model_config.quant_config.backend;

  if (fuse_gate_up_proj_) {
    gate_up_proj_ = std::make_shared<Linear>(fmt::format(layer_prefix + weight_name_format, "gate_up_proj"), ctx,
                                             linear_compute_backend);
  } else {
    gate_proj_ = std::make_shared<Linear>(fmt::format(layer_prefix + weight_name_format, "gate_proj"), ctx,
                                          linear_compute_backend);
    up_proj_ = std::make_shared<Linear>(fmt::format(layer_prefix + weight_name_format, "up_proj"), ctx,
                                        linear_compute_backend);
  }

  down_proj_ = std::make_shared<Linear>(fmt::format(layer_prefix + weight_name_format, "down_proj"), ctx,
                                        linear_compute_backend);

  if (!fuse_silu_mul_) {
    silu_mul_ = std::make_shared<SiluMul>(ctx);
  }
}

void SharedExpertLayer::SetParallelBuffers(TensorBuffer* inter_buffer, TensorBuffer* output_buffer) {
  parallel_inter_buffer_ = inter_buffer;
  parallel_output_buffer_ = output_buffer;
}

Status SharedExpertLayer::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                  std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                  ForwardingContext& forwarding_context, const LayerExecContext& exec_ctx) {
  // When exec_ctx has stream override and parallel buffers are set, use independent buffers.
  // Otherwise, fall back to shared hidden_buffer_1 from ForwardingBuffers.
  const bool use_parallel_buffers = exec_ctx.HasStreamOverride() && parallel_inter_buffer_ && parallel_output_buffer_;
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);

  if (use_parallel_buffers) {
    KLLM_LOG_INFO << "Using parallel buffers for SharedExpertLayer";
    // Use independent buffers for multi-stream parallel execution.
    CREATE_BUFFER_SCOPE(inter_tensors, parallel_inter_buffer_);
    CREATE_BUFFER_SCOPE(output_tensors, parallel_output_buffer_);
    inter_tensors[0].shape = hidden_buffer_tensors_0[0].shape;
    output_tensors[0].shape = hidden_buffer_tensors_0[0].shape;

    if (fuse_gate_up_proj_) {
      STATUS_CHECK_RETURN(gate_up_proj_->Forward(hidden_buffer_tensors_0[0], inter_tensors[0], exec_ctx));
      // std::swap(inter_tensors, hidden_buffer_tensors_0);
      if (!fuse_silu_mul_) {
        STATUS_CHECK_RETURN(silu_mul_->Forward(inter_tensors[0], output_tensors, exec_ctx));
        // std::swap(hidden_buffer_tensors_0, inter_tensors);
      }
    } else {
      STATUS_CHECK_RETURN(gate_proj_->Forward(hidden_buffer_tensors_0[0], inter_tensors[0], exec_ctx));
      STATUS_CHECK_RETURN(up_proj_->Forward(hidden_buffer_tensors_0[0], output_tensors[0], exec_ctx));
      // std::swap(inter_tensors, hidden_buffer_tensors_0);
      STATUS_CHECK_RETURN(silu_mul_->Forward(inter_tensors[0], output_tensors[0], inter_tensors, exec_ctx));
    }

    // down_proj output stays in output_tensors (parallel_output_buffer_).
    // In parallel mode, caller reads SharedExpert output via GetParallelOutputBuffer().
    // STATUS_CHECK_RETURN(down_proj_->Forward(hidden_buffer_tensors_1, output_tensors, exec_ctx));
    STATUS_CHECK_RETURN(down_proj_->Forward(inter_tensors[0], output_tensors[0], exec_ctx));

  } else {
    // Default path: use shared hidden_buffer_1 (same as TwoLayeredFFN)
    // CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);

    if (fuse_gate_up_proj_) {
      STATUS_CHECK_RETURN(gate_up_proj_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0]));
      std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
      if (!fuse_silu_mul_) {
        STATUS_CHECK_RETURN(silu_mul_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_1));
        std::swap(hidden_buffer_tensors_0, hidden_buffer_tensors_1);
      }
    } else {
      auto& gated_buffer_ = reduce_buffer_tensors;
      STATUS_CHECK_RETURN(gate_proj_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0]));
      STATUS_CHECK_RETURN(up_proj_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0]));
      std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
      STATUS_CHECK_RETURN(silu_mul_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], hidden_buffer_tensors_0));
    }

    // down_proj MatMul
    if (forwarding_context.GetModelCommunicator() && !enable_full_shared_expert_) {
      STATUS_CHECK_RETURN(down_proj_->Forward(hidden_buffer_tensors_0[0], reduce_buffer_tensors[0]));
    } else {
      STATUS_CHECK_RETURN(down_proj_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0]));
      std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
    }
  }

  return Status();
}

}  // namespace ksana_llm
