/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/layer_exec_context.h"
#include "models/base/buffer_manager.h"
#include "models/base/forwarding_context.h"
#include "modules/basic/linear.h"
#include "modules/basic/silu_mul.h"
#include "modules/basic/situ_mul.h"

namespace ksana_llm {

/**
 * @class SharedExpertLayer
 * @brief Independent shared expert layer for MoE models.
 *
 * This class implements the shared expert FFN (gate_up_proj + silu_mul + down_proj)
 * as an independent layer with its own Linear operators. Unlike TwoLayeredFFN,
 * it supports LayerExecContext to allow future multi-stream parallel execution
 * with routed experts.
 *
 * The shared expert can optionally use a separate workspace manager to avoid
 * conflicts when running concurrently with the MoE layer on a different stream.
 */
class SharedExpertLayer {
 public:
  /**
   * @brief Construct a SharedExpertLayer.
   *
   * @param layer_idx The layer index in the model.
   * @param creation_context The layer creation context containing factories and resources.
   * @param model_creation_config The model creation configuration.
   * @param weight_name_format The weight name format string (e.g. ".mlp.shared_expert.{}.weight").
   */
  SharedExpertLayer(int layer_idx, LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config,
                    const std::string& weight_name_format);

  ~SharedExpertLayer() = default;

  /**
   * @brief Set independent buffers for multi-stream parallel execution.
   *
   * When SharedExpert runs on a secondary stream in parallel with MoE,
   * it cannot share hidden_buffer_1 and reduce_buffer with the primary stream.
   * Call this method to provide independent intermediate and output buffers.
   *
   * @param inter_buffer Intermediate buffer for gate_up_proj output / silu_mul.
   *                     Shape: [max_token_num, 2 * shared_expert_inter_size / tp].
   * @param output_buffer Output buffer for down_proj result.
   *                      Shape: [max_token_num, hidden_size].
   */
  void SetParallelBuffers(TensorBuffer* inter_buffer, TensorBuffer* output_buffer);

  // Get the parallel output buffer (for caller to read SharedExpert result in parallel mode).
  TensorBuffer* GetParallelOutputBuffer() const { return parallel_output_buffer_; }

  /**
   * @brief Forward pass for the shared expert layer.
   *
   * @param hidden_buffer_tensors_0 Input hidden states.
   * @param reduce_buffer_tensors Output buffer for the shared expert result.
   * @param is_multi_token_forward Indicates if processing multiple tokens (prefill).
   * @param forwarding_context The forwarding context with runtime information.
   * @param exec_ctx Optional execution context override (stream, workspace).
   *                 Default {} = use primary stream and default workspace.
   * @return Status indicating success or failure.
   */
  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                 const bool is_multi_token_forward, ForwardingContext& forwarding_context,
                 const LayerExecContext& exec_ctx = {});

 private:
  bool fuse_gate_up_proj_ = false;
  bool fuse_silu_mul_ = false;
  bool enable_full_shared_expert_ = false;

  std::shared_ptr<Linear> gate_up_proj_;
  std::shared_ptr<Linear> gate_proj_;
  std::shared_ptr<Linear> up_proj_;
  std::shared_ptr<Linear> down_proj_;
  std::shared_ptr<SiluMul> silu_mul_;
  std::shared_ptr<SituMul> situ_mul_;
  bool use_situ_ = false;

  // Independent buffers for multi-stream parallel (optional, set via SetParallelBuffers).
  // When non-null, Forward uses these instead of shared hidden_buffer_1.
  TensorBuffer* parallel_inter_buffer_ = nullptr;   // [max_T, 2 * inter_size / tp]
  TensorBuffer* parallel_output_buffer_ = nullptr;  // [max_T, hidden_size]
};

}  // namespace ksana_llm
