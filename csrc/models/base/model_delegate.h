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
struct ForwardRequest;
class HiddenUnitDeviceBuffer;

// Narrow delegation interface for speculative decoders (e.g., MtpDecoder)
// to invoke model operations without depending on the concrete CommonModel type.
//
// Implemented by CommonModel via multiple inheritance.
// MtpDecoder receives a ModelDelegate* at construction and delegates model-level
// operations (embed, forward, LmHead, hidden buffer save) through this interface,
// keeping the decoder independent of the specific model implementation.
class ModelDelegate {
 public:
  virtual ~ModelDelegate() = default;

  // Expose the forwarding context for MTP buffer and metadata access.
  virtual ForwardingContext* GetForwardingContext() { return nullptr; }

  // Set the hidden unit buffer (shared across main forward and MTP steps).
  virtual void SetHiddenUnitBufferTensors(HiddenUnitDeviceBuffer* hidden_unit) {}

  // Get the residual buffer tensors for the current forward step.
  virtual std::vector<Tensor>& GetHiddenUnitBufferTensors(ForwardingContext& forwarding_context) {
    static std::vector<Tensor> empty;
    return empty;
  }

  // GPU embedding lookup for draft tokens.
  virtual Status EmbedTokensUseGpu(Tensor&, ForwardingContext&) { return Status(); }

  // Save hidden states for speculative decoding (next step input).
  // Called BEFORE MHC squeeze for models that save unnormed hidden.
  virtual void SaveNextStepHiddenBuffer(ForwardingContext& forwarding_context) {}

  // Save hidden states for speculative decoding, called AFTER MHC squeeze + norm.
  // Used by models with return_hidden_states_before_norm=false.
  virtual void SaveNextStepNormedHiddenBuffer(ForwardingContext& forwarding_context) {}

  // Post-LmHead: assemble -> lm_head -> reduce_sum -> all_gather -> cast -> logits.
  virtual Status LmHeadPost(ForwardingContext& forwarding_context, std::shared_ptr<BaseWeight>& base_weight,
                            std::vector<ForwardRequest>& forward_reqs, bool assemble_already_done,
                            std::vector<Tensor>& hidden_buffer_tensors_0) {
    return Status();
  }
};

}  // namespace ksana_llm
