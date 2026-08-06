/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "utils/status.h"

namespace ksana_llm {

// Forward declarations
class Tensor;
class ForwardingContext;
class BaseWeight;

// Wide-boundary interface for a draft model execution engine.
// Encapsulates a complete forward inference pipeline for a single token
// generation step: embedding → decoder layer → MHC → norm → lm_head.
//
// Implementations are model-specific (e.g. MTP layers, standalone draft models)
// and are injected into MtpDecoder or other GPU-type speculative decoders.
//
// Three-step pipeline: MtpDecoder calls each step directly.
// without going through CommonModel::Forward, achieving complete decoupling.
class DraftModelInterface {
 public:
  virtual ~DraftModelInterface() = default;

  // Step 1: Embedding — embed the draft token and concat with the previous
  // step's hidden state. Uses the model's embedding weights from BaseWeight.
  // Input:  residual_buffer — hidden states from the previous step
  //         base_weight — model weights (for "model.embed_tokens.weight")
  //         forwarding_context — buffer management and input metadata
  // Output: residual_buffer — expanded hidden states ready for layer forward
  virtual Status ForwardEmbedding(std::vector<Tensor>& residual_buffer, std::shared_ptr<BaseWeight>& base_weight,
                                  ForwardingContext& forwarding_context) {
    return Status();
  }

  // Step 2: Decoder layer — execute one draft-model decoder layer forward
  // step. This replaces the original single Forward() method.
  // Input:  residual_buffer — hidden states from embedding step
  //         forwarding_context — buffer management
  // Output: residual_buffer — hidden states ready for MHC/norm/lm_head
  virtual Status ForwardLayer(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) = 0;

  // Step 3: MHC squeeze → final norm → LmHead → logits.
  // Uses the draft model's own final_mhc and shared_head_norm (injected at
  // construction), plus the model's lm_head weight from BaseWeight.
  // Input:  residual_buffer — hidden states from decoder layer
  //         base_weight — model weights (for lm_head)
  //         forwarding_context — buffer management
  // Output: residual_buffer — logits ready for sampling
  virtual Status ForwardMhcNormLmHead(std::vector<Tensor>& residual_buffer, std::shared_ptr<BaseWeight>& base_weight,
                                      ForwardingContext& forwarding_context) {
    return Status();
  }
};

}  // namespace ksana_llm