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

  // Radix path: scores, row_starts, row_ends, page_table, optional batch_indices.
  Status ForwardRadix(const Tensor* scores, const Tensor* row_starts, const Tensor* row_ends, const Tensor* page_table,
                      Tensor* output, const Tensor* batch_indices = nullptr, const LayerExecContext& exec_ctx = {});
  // V2 path: workspace_stride_bytes is workspace.shape[1].
  Status ForwardV2(const Tensor* scores, const Tensor* seq_lens, const Tensor* page_table, const Tensor* workspace,
                   int64_t workspace_stride_bytes, const Tensor* metadata, Tensor* output,
                   const LayerExecContext& exec_ctx = {});

  void SetUseV2(bool use_v2) { use_v2_ = use_v2; }

 private:
  uint32_t k_ = 512;
  uint32_t page_bits_ = 4;
  bool use_v2_ = true;
};

}  // namespace ksana_llm
