/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>
#include <memory>
#include <vector>

#include "layers/layer_exec_context.h"
#include "models/base/layer_creation_context.h"

namespace ksana_llm {

class Linear {
 public:
  // Disable a default constructor
  Linear(const std::string& weight_name, const LayerCreationContext& creation_context,
         const LinearComputeBackend& linear_compute_backend, const bool skip_quant = false);

  ~Linear() = default;

  Status Forward(Tensor input_tensor, std::vector<Tensor>& output_tensors, const LayerExecContext& exec_ctx = {});

  // TODO(robertyuan): Remove later
  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                 const LayerExecContext& exec_ctx = {});

 protected:
  std::shared_ptr<BaseLayer> proj_layer_;
  Tensor weight_;
};

}  // namespace ksana_llm
