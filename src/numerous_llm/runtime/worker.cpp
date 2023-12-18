/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/worker.h"

namespace numerous_llm {

Status Worker::Execute(Context& ctx, const InferStage stage, const int worker_id,
                       const std::vector<TensorMap*>& input_tensor_maps, std::vector<TensorMap*>& output_tensor_maps) {
  CUDA_CHECK(cudaSetDevice(worker_id));

  switch (stage) {
    case InferStage::STAGE_CONTEXT:
      NLLM_LOG_INFO << "ContextDecode infer on work_id: " << worker_id;
      base_model_ptr_->ContextDecode(base_weight_ptr_, input_tensor_maps, output_tensor_maps);
      break;
    case InferStage::STATE_DECODE:
      NLLM_LOG_INFO << "Decode infer on work_id: " << worker_id;
      base_model_ptr_->Decode(base_weight_ptr_, input_tensor_maps, output_tensor_maps);
      break;
    default:
      throw std::invalid_argument("Unknown infer stage");
      break;
  }

  return Status();
}

}  // namespace numerous_llm
