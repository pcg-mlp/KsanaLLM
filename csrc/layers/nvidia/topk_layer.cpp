/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/topk_layer.h"

#include "csrc/kernels/nvidia/logits_topk/logits_topk.h"
#include "csrc/kernels/nvidia/others/sglang/main/topk/topk.h"
#include "csrc/kernels/nvidia/others/sglang/main/topk/topk_wrapper.h"
#include "device/nvidia/cuda_utils.h"
#include "utils/logger.h"

namespace ksana_llm {

using llm_kernels::nvidia::topk::kTopK1024;
using llm_kernels::nvidia::topk::kTopK512;
using llm_kernels::nvidia::topk::TopKWrapper;

Status TopKLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                       std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);

  // Parameter extraction (positional).
  int param_idx = 0;
  k_ = std::any_cast<uint32_t>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(k_ == kTopK512 || k_ == kTopK1024 || k_ == 2048,
                       fmt::format("TopKLayer: unsupported k={} (must be 512, 1024, or 2048)", k_));
  const uint32_t block_size = std::any_cast<uint32_t>(parameters[param_idx++]);
  KLLM_CHECK(block_size > 0 && (block_size & (block_size - 1)) == 0);
  page_bits_ = static_cast<uint32_t>(__builtin_ctz(block_size));

  return Status();
}

// ---------------------------------------------------------------------------
// Radix path (topk.cuh / topk_1024.cuh / InvokeFastTopK)
// ---------------------------------------------------------------------------
Status TopKLayer::ForwardRadix(const Tensor* scores, const Tensor* row_starts, const Tensor* row_ends,
                               const Tensor* page_table, Tensor* output, const Tensor* batch_indices,
                               const LayerExecContext& exec_ctx) {
  const uint32_t num_works = static_cast<uint32_t>(scores->shape[0]);
  const int64_t score_stride = static_cast<int64_t>(scores->shape[1]);
  const uint32_t num_batches = static_cast<uint32_t>(page_table->shape[0]);
  const int64_t page_table_stride = static_cast<int64_t>(page_table->shape[1]);

  int32_t* raw_indices_ptr = nullptr;
  auto stream = context_->device->GetPrimaryComputeStream().Get();

  if (k_ == 2048) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeFastTopK(scores->GetPtr<float>(),          // logits
                                                              row_starts->GetPtr<int32_t>(),    // rowStarts
                                                              row_ends->GetPtr<int32_t>(),      // rowEnds
                                                              output->GetPtr<int32_t>(),        // indices
                                                              static_cast<int64_t>(num_works),  // numRows
                                                              score_stride,                     // stride0
                                                              static_cast<int64_t>(1),          // stride1
                                                              stream));
  } else {
    const int32_t* batch_indices_ptr = nullptr;
    uint32_t extend_len;
    if (batch_indices != nullptr) {
      batch_indices_ptr = batch_indices->GetPtr<int32_t>();
      extend_len = 0;
    } else {
      extend_len = num_works / num_batches;
    }

    if (k_ == kTopK512) {
      llm_kernels::nvidia::topk::launch_topk512(scores->GetPtr<float>(),        // scores
                                                row_starts->GetPtr<int32_t>(),  // row_starts
                                                row_ends->GetPtr<int32_t>(),    // row_ends
                                                page_table->GetPtr<int32_t>(),  // page_table
                                                output->GetPtr<int32_t>(),      // page_indices_out
                                                raw_indices_ptr,                // raw_indices_out
                                                score_stride,                   // score_stride
                                                page_table_stride,              // page_table_stride
                                                num_works,                      // num_works
                                                page_bits_,                     // page_bits
                                                batch_indices_ptr,              // batch_indices
                                                extend_len,                     // extend_len
                                                stream);                        // stream
    } else {
      // k_ == kTopK1024
      llm_kernels::nvidia::topk::launch_topk1024(scores->GetPtr<float>(),        // scores
                                                 row_starts->GetPtr<int32_t>(),  // row_starts
                                                 row_ends->GetPtr<int32_t>(),    // row_ends
                                                 page_table->GetPtr<int32_t>(),  // page_table
                                                 output->GetPtr<int32_t>(),      // page_indices_out
                                                 raw_indices_ptr,                // raw_indices_out
                                                 score_stride,                   // score_stride
                                                 page_table_stride,              // page_table_stride
                                                 num_works,                      // num_works
                                                 page_bits_,                     // page_bits
                                                 batch_indices_ptr,              // batch_indices
                                                 extend_len,                     // extend_len
                                                 stream);                        // stream
    }
  }

  output->shape = {num_works, k_};
  output->dtype = DataType::TYPE_INT32;
  return Status();
}

// ---------------------------------------------------------------------------
// Multi-path v2 (topk_v2.cuh)
// ---------------------------------------------------------------------------
Status TopKLayer::ForwardV2(const Tensor* scores, const Tensor* seq_lens, const Tensor* page_table,
                            const Tensor* workspace, int64_t workspace_stride_bytes, const Tensor* metadata,
                            Tensor* output, const LayerExecContext& exec_ctx) {
  const uint32_t num_works = static_cast<uint32_t>(scores->shape[0]);
  const int64_t score_stride = static_cast<int64_t>(scores->shape[1]);
  const uint32_t num_batches = static_cast<uint32_t>(page_table->shape[0]);
  const int64_t page_table_stride = static_cast<int64_t>(page_table->shape[1]);
  const uint32_t extend_len = num_works / num_batches;

  auto stream = context_->device->GetPrimaryComputeStream().Get();

  TopKWrapper<float, int32_t>::TransformV2(seq_lens->GetPtr<void>(), scores->GetPtr<void>(), page_table->GetPtr<void>(),
                                           output->GetPtr<void>(), workspace->GetPtr<void>(), metadata->GetPtr<void>(),
                                           score_stride, page_table_stride, workspace_stride_bytes, num_works,
                                           page_bits_, /*max_seq_len*/ score_stride,
                                           /*static_cluster_threshold=*/0, nullptr, extend_len, k_, stream);

  output->shape = {num_works, k_};
  output->dtype = DataType::TYPE_INT32;
  return Status();
}

}  // namespace ksana_llm
