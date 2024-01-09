/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/samplers/sampler.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/memory_utils.h"

namespace numerous_llm {

Sampler::Sampler(int rank) {
  rank_ = rank;
  topk_sampling_ = new TopkSampling();

  // need to allocate device buffer for sampling
  Tensor output_token_tensor;
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(sizeof(uint32_t), output_token_block_id_);
  void* device_output_token_ptr;
  GetBlockManager()->GetContiguousPtr(output_token_block_id_, device_output_token_ptr);
  device_output_token_ = static_cast<uint32_t*>(device_output_token_ptr);

  Tensor offset_tensor;
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(sizeof(uint32_t), offset_block_id_);
  void* device_offset_ptr;
  GetBlockManager()->GetContiguousPtr(offset_block_id_, device_offset_ptr);
  device_offset_ = static_cast<uint32_t*>(device_offset_ptr);
}

Sampler::~Sampler() {
  delete topk_sampling_;

  // free device buffer of output tokens
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->FreeContiguous(output_token_block_id_);
  GetBlockManager()->FreeContiguous(offset_block_id_);
}

Status Sampler::Sampling(std::vector<SamplingRequest>& sampling_reqs, cudaStream_t& stream) {
  if (rank_ == 0) {
    // NOTE(catheywang): Sampling_reqs means different batches,
    // and each sampling_req contains a batch of logits_buf(vector of logits_buf is for different gpus).
    // All batches' logits_buf are the same but with different offsets.
    // While each sampling_seq has its own sampling_config, we can't sample sampling_seqs at once
    // Loop of sampling_reqs and sampling one batch at a time is a temporary solution.
    for (auto& sampling_req : sampling_reqs) {
      const ModelConfig* model_config = sampling_req.model_config;
      const SamplingConfig* sampling_config = sampling_req.sampling_config;
      const float* logits = sampling_req.logits_buf[rank_] + sampling_req.logits_offset * model_config->vocab_size;
      CUDA_CHECK(cudaMemsetAsync(device_offset_, 0, sizeof(uint32_t), stream));
      std::vector<int>* output_tokens = sampling_req.output_tokens;
      if (sampling_config->beam_width == 1) {
        STATUS_CHECK_RETURN(topk_sampling_->Forward(logits, device_offset_, device_output_token_, sampling_config,
                                                    model_config, stream));
      } else {
        return Status(RET_INVALID_ARGUMENT, "sampling for beam_width > 1 not implemented");
      }
      // copy result
      CUDA_CHECK(
          cudaMemcpyAsync(&host_output_token_, device_output_token_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
      output_tokens->push_back(static_cast<int>(host_output_token_));
    }
  }
  return Status();
}

}  // namespace numerous_llm
