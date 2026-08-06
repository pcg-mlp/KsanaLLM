/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

namespace ksana_llm {

class Tensor;
struct ForwardRequest;

// Abstract interface for external gather index builders.
// Injected into ModelInput by speculative decoders (e.g., MtpDecoder) to override
// the default MTP hidden state gathering mechanism. Concrete implementations live
// in runtime/speculative/ and implement decoder-specific indexing logic.
class GatherIndexBuilderInterface {
 public:
  virtual ~GatherIndexBuilderInterface() = default;

  // Record each request's starting position in the main forward batch.
  virtual void UpdateMainForward(const std::vector<ForwardRequest>& forward_reqs) = 0;

  // Build gather indices from the positions recorded in UpdateMainForward.
  virtual void BuildGatherIndices(const std::vector<ForwardRequest>& forward_reqs) = 0;

  // Return the gather index tensor used by AssembleTokensHiddenLayer during MTP forward.
  virtual Tensor* GetGatherTensor() = 0;
};

}  // namespace ksana_llm
