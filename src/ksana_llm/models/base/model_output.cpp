/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_output.h"

namespace ksana_llm {

ModelOutput::ModelOutput(size_t max_batch_size, size_t vocab_size, int rank, std::shared_ptr<Context> context)
    : rank_(rank), context_(context), max_batch_size_(max_batch_size), vocab_size_(vocab_size) {
  STATUS_CHECK_FAILURE(CreateTensor(logits_tensor, {max_batch_size_, vocab_size_}, TYPE_FP32, rank_, MEMORY_DEVICE));

  EventCreateWithFlags(&compute_ready_event, EVENT_DISABLE_TIMING);
}

ModelOutput::~ModelOutput() {
  STATUS_CHECK_FAILURE(DestroyTensor(logits_tensor, rank_));

  EventDestroy(compute_ready_event);
}

void ModelOutput::CopyToLogistBuffer(const size_t batch_size, std::vector<ForwardRequest>& forward_reqs,
                                     std::vector<Tensor>& logits_float) {
  EventRecord(compute_ready_event, context_->GetComputeStreams()[rank_]);
  StreamWaitEvent(context_->GetD2DStreams()[rank_], compute_ready_event);
  // Copy to logits buf
  float* logits_ptr = logits_float[0].GetPtr<float>();
  float* logits_dst = forward_reqs[0].logits_buf[rank_] + forward_reqs[0].logits_offset * vocab_size_;
  MemcpyAsync(logits_dst, logits_ptr, batch_size * vocab_size_ * sizeof(float), MEMCPY_DEVICE_TO_DEVICE,
              context_->GetD2DStreams()[rank_]);
  StreamSynchronize(context_->GetD2DStreams()[rank_]);
}

}  // namespace ksana_llm
