/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/mla_output_gate_layer.h"

#include <fmt/core.h>

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/others/sglang/main/kimi_k3/mla_output_gate.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

Status MlaOutputGateLayer::Forward(const Tensor* x, const Tensor* gate, Tensor* output,
                                   const LayerExecContext& exec_ctx) {
  KLLM_CHECK_WITH_INFO(inter_data_type_ == DataType::TYPE_BF16, "MlaOutputGateLayer only supports BF16.");
  KLLM_CHECK_WITH_INFO(x->shape == gate->shape, fmt::format("MlaOutputGateLayer: x shape {} must match gate shape {}",
                                                            Vector2Str(std::vector<size_t>(x->shape)),
                                                            Vector2Str(std::vector<size_t>(gate->shape))));

  const size_t num_elements = x->GetElementNumber();
  llm_kernels::nvidia::mla_output_gate(x->GetPtr<void>(), gate->GetPtr<void>(), output->GetPtr<void>(), num_elements,
                                       GetComputeStream(exec_ctx));

  output->shape = x->shape;
  output->dtype = x->dtype;
  return Status();
}

}  // namespace ksana_llm
