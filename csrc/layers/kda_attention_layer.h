/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <optional>

#include "layers/base_layer.h"

namespace ksana_llm {

// Tensors shared by the two model-facing KDA execution modes. Keeping this
// contract free of CUDA types lets model code select KDA without depending on
// the NVIDIA kernel ABI; backend-only verify metadata remains in the wrapper.
struct KdaAttentionInputs {
  // Projection outputs and per-layer weights. qkv and beta are BF16 model
  // activations; conv_weight, a_log and dt_bias are FP32 kernel parameters.
  const Tensor* qkv{nullptr};
  const Tensor* forget_gate{nullptr};
  const Tensor* beta{nullptr};
  const Tensor* output_gate{nullptr};
  const Tensor* conv_weight{nullptr};
  const Tensor* a_log{nullptr};
  const Tensor* dt_bias{nullptr};
  const Tensor* norm_weight{nullptr};

  Tensor* conv_state{nullptr};
  Tensor* ssm_state{nullptr};
  const Tensor* state_indices{nullptr};

  // Direct Cache stores every local KDA layer in one request block. These
  // element pitches let the backend address a layer view without changing the
  // legacy dense kernel ABI; zero retains the dense fallback.
  int conv_state_slot_stride{0};
  int ssm_state_slot_stride{0};
};

struct KdaPrefillInputs {
  KdaAttentionInputs common;
  int max_seq_len{0};
  const Tensor* cu_seqlens{nullptr};
  const Tensor* chunk_indices{nullptr};
  const Tensor* chunk_offsets{nullptr};
  const Tensor* has_initial_state{nullptr};
};

// Decode is deliberately the ordinary one-token-per-sequence path. Target
// verify has a different lifecycle and remains a lower-level wrapper contract
// until the model and scheduler support speculative KDA state checkpoints.
struct KdaDecodeInputs {
  KdaAttentionInputs common;
};

struct KdaAttentionLayerParameters : public BaseLayerParameters {
  uint32_t num_heads{0};
  uint32_t head_dim{0};
  std::optional<float> gate_lower_bound{std::nullopt};
  float norm_eps{1e-5f};
};

class KdaAttentionLayer : public BaseLayer {
 public:
  Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
              std::shared_ptr<Context> context, int rank) override;

  size_t GetWorkspaceSize() override;

  Status ForwardPrefill(const KdaPrefillInputs& inputs, Tensor* output, const LayerExecContext& exec_ctx = {});
  Status ForwardDecode(const KdaDecodeInputs& inputs, Tensor* output, const LayerExecContext& exec_ctx = {});

 private:
  uint32_t num_heads_{0};
  uint32_t head_dim_{0};
  size_t max_step_token_num_{0};
  size_t max_batch_size_{0};
  std::optional<float> gate_lower_bound_{std::nullopt};
  float norm_eps_{1e-5f};
};

}  // namespace ksana_llm
