/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/topk_softplus_sqrt_layer.h"

#include <cstdlib>  // std::getenv

#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/sglang_gate_dtype.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/topk_softplus_sqrt_sglang.h"
#include "csrc/kernels/nvidia/others/vllm/main/topk_softplus_sqrt/topk_softplus_sqrt.h"

namespace ksana_llm {

Status TopKSoftplusSqrtLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                   std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  params_ = std::any_cast<const TopKSoftplusSqrtLayerParameters>(parameters.front());

  KLLM_CHECK_WITH_INFO(params_.topk > 0, "TopKSoftplusSqrtLayer: topk must be > 0");
  KLLM_CHECK_WITH_INFO(llm_kernels::nvidia::kTopkSoftplusSqrtSupportedNumExperts.count(params_.num_experts) > 0,
                       fmt::format("TopKSoftplusSqrtLayer: unsupported num_experts {}", params_.num_experts));
  KLLM_CHECK_WITH_INFO(!(params_.use_hash && params_.use_correction_bias),
                       "TopKSoftplusSqrtLayer: use_hash && use_correction_bias is invalid (hash path ignores bias)");
  KLLM_CHECK_WITH_INFO(
      params_.topk_ids_dtype == TYPE_INT32 || params_.topk_ids_dtype == TYPE_INT64,
      fmt::format("TopKSoftplusSqrtLayer: topk_ids_dtype must be INT32 or INT64, got {}", params_.topk_ids_dtype));

  KLLM_LOG_INFO << fmt::format("Rank[{}] TopKSoftplusSqrtLayer Init with {}", rank_, params_.ToString());

  const char* env = std::getenv("KSANA_MOE_GATE_BACKEND");
  if (env != nullptr && std::string(env) == "sglang") {
    backend_ = SqrtSoftplusBackend::kSglang;
  } else if (env != nullptr && std::string(env) == "vllm") {
    backend_ = SqrtSoftplusBackend::kVllm;
  } else {
    backend_ = SqrtSoftplusBackend::kHybrid;
  }

  const bool will_use_sglang =
      (backend_ == SqrtSoftplusBackend::kSglang) || (backend_ == SqrtSoftplusBackend::kHybrid && params_.use_hash);
  if (will_use_sglang) {
    const bool ok = (params_.num_experts == 256 || params_.num_experts == 384) &&
                    (params_.topk == 6 || params_.topk == 8) && (params_.topk_ids_dtype == TYPE_INT32);
    KLLM_CHECK_WITH_INFO(ok,
                         fmt::format("sglang gate kernel requires num_experts in {{256,384}}, "
                                     "topk in {{6,8}}, topk_ids_dtype=INT32; got num_experts={}, topk={}, ids_dtype={}",
                                     params_.num_experts, params_.topk, params_.topk_ids_dtype));
  }

  const char* backend_str = (backend_ == SqrtSoftplusBackend::kHybrid)   ? "hybrid"
                            : (backend_ == SqrtSoftplusBackend::kSglang) ? "sglang"
                                                                         : "vllm";
  KLLM_LOG_INFO << fmt::format("Rank[{}] TopKSoftplusSqrtLayer backend={}, use_hash={}", rank_, backend_str,
                               params_.use_hash);
  return Status();
}

size_t TopKSoftplusSqrtLayer::GetWorkspaceSize() {
  const bool uses_sglang =
      (backend_ == SqrtSoftplusBackend::kSglang) || (backend_ == SqrtSoftplusBackend::kHybrid && params_.use_hash);
  if (uses_sglang || params_.use_hash) return 0;
  return static_cast<size_t>(params_.max_token_num) * params_.topk * sizeof(int32_t);
}

Status TopKSoftplusSqrtLayer::Forward(const Tensor* router_logits, Tensor* topk_weights, Tensor* topk_ids,
                                      const Tensor* aux1, const Tensor* aux2, const LayerExecContext& exec_ctx) {
  // Inputs:
  //   router_logits [N, num_experts], fp32/fp16/bf16
  //   if use_hash:
  //     aux1: input_ids [N], int32 or int64
  //     aux2: tid2eid   [vocab, topk], same int type as aux1
  //   elif use_correction_bias:
  //     aux1: correction_bias [num_experts], fp32
  //
  // Outputs:
  //   topk_weights [N, topk], fp32
  //   topk_ids     [N, topk], int32 or int64 (dtype from params_.topk_ids_dtype)

  const uint32_t num_tokens = static_cast<uint32_t>(router_logits->shape[0]);
  const std::vector<size_t> out_shape = {static_cast<size_t>(num_tokens), static_cast<size_t>(params_.topk)};
  topk_weights->shape = out_shape;
  topk_weights->dtype = TYPE_FP32;
  topk_ids->shape = out_shape;
  topk_ids->dtype = params_.topk_ids_dtype;

  if (num_tokens == 0) return Status();

  const void* input_ids = nullptr;
  const void* tid2eid = nullptr;
  const float* bias = nullptr;
  if (params_.use_hash) {
    KLLM_CHECK_WITH_INFO(aux1 != nullptr && aux2 != nullptr,
                         "TopKSoftplusSqrtLayer: use_hash requires aux1 (input_ids) and aux2 (tid2eid)");
    input_ids = aux1->GetPtr<void>();
    tid2eid = aux2->GetPtr<void>();
  } else if (params_.use_correction_bias) {
    KLLM_CHECK_WITH_INFO(aux1 != nullptr, "TopKSoftplusSqrtLayer: use_correction_bias requires aux1");
    bias = aux1->GetPtr<float>();
  }

  const bool use_sglang_path =
      (backend_ == SqrtSoftplusBackend::kSglang) || (backend_ == SqrtSoftplusBackend::kHybrid && params_.use_hash);
  if (use_sglang_path) {
    KLLM_CHECK_WITH_INFO(
        router_logits->dtype == TYPE_FP32 || router_logits->dtype == TYPE_BF16,
        fmt::format("sglang backend requires fp32 or bf16 router_logits, got dtype={}", router_logits->dtype));
    const auto sglang_input_dtype = (router_logits->dtype == TYPE_FP32)
                                        ? llm_kernels::nvidia::sglang_gate::GateInputDtype::kFloat32
                                        : llm_kernels::nvidia::sglang_gate::GateInputDtype::kBFloat16;
    llm_kernels::nvidia::sglang_gate::InvokeTopkSoftplusSqrtSglang(
        router_logits->GetPtr<void>(), sglang_input_dtype, topk_weights->GetPtr<void>(), topk_ids->GetPtr<void>(),
        static_cast<int>(num_tokens), static_cast<int>(params_.num_experts), static_cast<int>(params_.topk),
        params_.renormalize, params_.routed_scaling_factor, bias, input_ids, tid2eid,
        context_->device->GetPrimaryComputeStream().Get());
  } else {
    void* token_expert_indices = params_.use_hash ? nullptr : workspace_buffer_->GetPtr<void>();
    llm_kernels::nvidia::InvokeTopkSoftplusSqrt(
        router_logits->GetPtr<void>(), topk_weights->GetPtr<void>(), topk_ids->GetPtr<void>(), token_expert_indices,
        static_cast<int>(num_tokens), static_cast<int>(params_.num_experts), static_cast<int>(params_.topk),
        params_.renormalize, static_cast<double>(params_.routed_scaling_factor), bias, input_ids, tid2eid,
        DataTypeToScalarTypeMap.at(router_logits->dtype), DataTypeToScalarTypeMap.at(params_.topk_ids_dtype),
        context_->device->GetPrimaryComputeStream().Get());
  }
  return Status();
}

}  // namespace ksana_llm
