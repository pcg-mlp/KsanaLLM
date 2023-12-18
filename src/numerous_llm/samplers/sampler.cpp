/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/samplers/sampler.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Status Sampler::Sampling(std::vector<std::shared_ptr<InferRequest>> &reqs) {
  NLLM_LOG_INFO << "llm sampler invoked.";

  for (std::shared_ptr<InferRequest> &req : reqs) {
    // TODO(karlluo): just a fake result for scheduler output result
    if (req->infer_stage == InferStage::STAGE_CONTEXT) {
      int block_id;
      Singleton<BlockManager>::GetInstance()->AllocateContiguous(1024, block_id);
      Tensor *t =
          new Tensor(MemoryDevice::MEMORY_CPU, StorageType::STORAGE_CONTIGUOUS, DataType::TYPE_FP32, {1}, {block_id});
      req->output_tensor_map.Insert("output", *t);
    }
  }

  return Status();
}

}  // namespace numerous_llm
