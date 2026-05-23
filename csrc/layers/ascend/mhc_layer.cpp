/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/mhc_layer.h"

#include "utils/logger.h"

namespace ksana_llm {

Status MhcLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}

size_t MhcLayer::GetMhcPreNumSplits(int token_num, int hc_hidden_dim) { return 0; }

Status MhcLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("MhcLayer not implement in Ascend.");
}

}  // namespace ksana_llm
