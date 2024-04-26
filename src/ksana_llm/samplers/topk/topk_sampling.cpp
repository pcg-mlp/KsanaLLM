/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/samplers/topk/topk_sampling.h"
#include "ksana_llm/utils/logger.h"

#ifdef ENABLE_CUDA
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/samplers/samplingTopKKernels.h"
#endif

#include "ksana_llm/kernels/argmax.h"

#include <cstdint>

namespace ksana_llm {
TopkSampling::TopkSampling(size_t max_batch_size, size_t max_vocab_size, RandState* device_curandstates)
    : BaseSampling(max_batch_size, max_vocab_size) {
#ifdef ENABLE_CUDA
  float* logits = nullptr;
  tensorrt_llm::kernels::invokeBatchTopKSampling(nullptr, workspace_size_, logits, nullptr, nullptr, nullptr, nullptr,
                                                 nullptr, nullptr, nullptr, 1024, nullptr, 0, nullptr,
                                                 static_cast<int>(max_vocab_size), nullptr, nullptr, nullptr,
                                                 static_cast<int>(max_batch_size), 0, nullptr, false, false);

  NLLM_LOG_DEBUG << "TopkSampling workspace_size_ " << workspace_size_;

  GetBlockManager()->AllocateContiguous(workspace_size_ + sizeof(uint64_t) * max_batch_size, workspace_block_id_);
  GetBlockManager()->GetContiguousPtr(workspace_block_id_, workspace_);
  tensorrt_llm::kernels::invokeCurandBatchInitialize(device_curandstates, nullptr, max_batch_size,
                                                     static_cast<uint64_t*>(workspace_ + workspace_size_), 0);
#endif
}

TopkSampling::~TopkSampling() {
  if (workspace_size_ > 0) {
    GetBlockManager()->FreeContiguous(workspace_block_id_);
  }
}

Status TopkSampling::RunSampling(float* logits, const uint32_t* offsets, uint32_t* output_token,
                                 const SamplingConfig* sampling_config,
                                 SamplingDevideParameter sampling_devide_parameter, const ModelConfig* model_config,
                                 Stream& stream) {
  if (sampling_devide_parameter.device_topKs == nullptr) {
    ArgMax(logits, offsets, sampling_devide_parameter.bs, sampling_devide_parameter.vocab_size_padded, output_token,
           stream);
  } else {
#ifdef ENABLE_CUDA
    bool logitHasProbs = false;
    if (sampling_devide_parameter.device_temperatures || sampling_devide_parameter.device_topPs) {
      logitHasProbs = true;
      tensorrt_llm::kernels::invokeAddBiasSoftMax<float>(
          logits, nullptr, sampling_devide_parameter.device_temperatures, nullptr, nullptr, nullptr, nullptr,
          sampling_devide_parameter.bs, 0, 1, sampling_devide_parameter.vocab_size_padded,
          sampling_devide_parameter.vocab_size_padded, false, true, stream.Get());
    }
    tensorrt_llm::kernels::invokeBatchTopKSampling(
        workspace_, workspace_size_, logits, sampling_devide_parameter.device_output_tokens_ptrs, nullptr, nullptr,
        nullptr, nullptr, nullptr, sampling_devide_parameter.device_curandstates, sampling_devide_parameter.max_topK,
        sampling_devide_parameter.device_topKs, 1.0, sampling_devide_parameter.device_topPs,
        static_cast<int>(sampling_devide_parameter.vocab_size_padded), nullptr, nullptr, stream.Get(),
        static_cast<int>(sampling_devide_parameter.bs), 0, nullptr, false, logitHasProbs);
#else
    throw std::runtime_error("Not support topk in Ascend NPU.");
#endif
  }

  return Status();
}

}  // namespace ksana_llm
