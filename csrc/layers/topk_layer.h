/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>
#include <cstdint>
#include <vector>

#include "layers/base_layer.h"
#include "utils/tensor.h"

namespace ksana_llm {

constexpr size_t kWrokspaceIntsPerBatch = 2 + 1024 * 2;

constexpr size_t kTopkV2MetadataIntsPerBatch = 4;
// -----------------------------------------------------------------------------
// TopKLayer
// -----------------------------------------------------------------------------
class TopKLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx = {}) override;

 private:
  Status ForwardRadix(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);
  Status ForwardV2(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  void SetUseV2(bool use_v2) { use_v2_ = use_v2; }

 private:
  uint32_t k_ = 512;
  uint32_t page_bits_ = 4;
  bool use_v2_ = true;
};

}  // namespace ksana_llm
