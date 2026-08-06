/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "models/base/gather_index_builder_interface.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

// Forward declarations
class Context;
struct ForwardRequest;
class Tensor;

// Builds gather indices for MTP (Multi-Token Prediction) hidden state selection.
//
// Owned by MtpDecoder, which injects it into ModelInput via SetExternalGatherIdxBuilder()
// before the MTP forward step. This removes MTP-specific state (mtp_req_id_to_pos_,
// nextn_hidden_idx_uint64_tensor) from ModelInput.
//
// Usage:
//   1. Main forward:  UpdateMainForward(reqs)  — records each request's starting position
//   2. MTP forward:   BuildGatherIndices(reqs) — builds gather indices from stored positions
//
// The gather tensor (GetGatherTensor()) is then read by AssembleTokensHiddenLayer
// during DeepSeekV4MtpLayer::Forward.
class MtpGatherIndexBuilder : public GatherIndexBuilderInterface {
 public:
  MtpGatherIndexBuilder(const size_t max_token_num, std::shared_ptr<Context> context, int rank);
  ~MtpGatherIndexBuilder() = default;

  void UpdateMainForward(const std::vector<ForwardRequest>& forward_reqs) override;

  // Build gather indices from the positions recorded in UpdateMainForward.
  // Called from the MTP forward's ParseFromRequests path.
  void BuildGatherIndices(const std::vector<ForwardRequest>& forward_reqs) override;

  // Return the gather index tensor for use by AssembleTokensHiddenLayer.
  // Valid only after BuildGatherIndices has been called.
  Tensor* GetGatherTensor() override;

 private:
  size_t max_token_num_;
  std::shared_ptr<Context> context_;
  int rank_;

  // Maps request IDs to their starting position in the main forward batch.
  std::unordered_map<int64_t, size_t> mtp_req_id_to_pos_;

  // Device tensor holding the gather indices (UINT64).
  // Shape: [num_gather_entries]. Populated in BuildGatherIndices.
  Tensor nextn_hidden_idx_uint64_tensor_;
};

}  // namespace ksana_llm
