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

Status TopKLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (use_v2_) {
    return ForwardV2(input_tensors, output_tensors);
  }
  return ForwardRadix(input_tensors, output_tensors);
}

// ---------------------------------------------------------------------------
// Radix path (topk.cuh / topk_1024.cuh / InvokeFastTopK)
// ---------------------------------------------------------------------------
Status TopKLayer::ForwardRadix(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // Inputs:  [0]=scores [1]=row_starts [2]=row_ends [3]=page_table [4]=batch_indices(optional)
  // Outputs: [0]=page_indices
  const auto& scores = input_tensors[0];
  const int32_t* row_starts_ptr = input_tensors[1].GetPtr<int32_t>();
  const int32_t* row_ends_ptr = input_tensors[2].GetPtr<int32_t>();
  const auto& page_table = input_tensors[3];

  const uint32_t num_works = static_cast<uint32_t>(scores.shape[0]);
  const int64_t score_stride = static_cast<int64_t>(scores.shape[1]);
  const uint32_t num_batches = static_cast<uint32_t>(page_table.shape[0]);
  const int64_t page_table_stride = static_cast<int64_t>(page_table.shape[1]);

  auto& page_indices = output_tensors[0];
  int32_t* raw_indices_ptr = nullptr;

  auto stream = context_->device->GetPrimaryComputeStream().Get();

  if (k_ == 2048) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeFastTopK(scores.GetPtr<float>(),           // logits
                                                              row_starts_ptr,                   // rowStarts
                                                              row_ends_ptr,                     // rowEnds
                                                              page_indices.GetPtr<int32_t>(),   // indices
                                                              static_cast<int64_t>(num_works),  // numRows
                                                              score_stride,                     // stride0
                                                              static_cast<int64_t>(1),          // stride1
                                                              stream));
  } else {
    const int32_t* batch_indices_ptr = nullptr;
    uint32_t extend_len;
    if (input_tensors.size() > 4) {
      // Prefill path: batch_indices array gives per-token batch index
      batch_indices_ptr = input_tensors[4].GetPtr<int32_t>();
      extend_len = 0;
    } else {
      extend_len = num_works / num_batches;
    }

    if (k_ == kTopK512) {
      llm_kernels::nvidia::topk::launch_topk512(scores.GetPtr<float>(),          // scores
                                                row_starts_ptr,                  // row_starts
                                                row_ends_ptr,                    // row_ends
                                                page_table.GetPtr<int32_t>(),    // page_table
                                                page_indices.GetPtr<int32_t>(),  // page_indices_out
                                                raw_indices_ptr,                 // raw_indices_out
                                                score_stride,                    // score_stride
                                                page_table_stride,               // page_table_stride
                                                num_works,                       // num_works
                                                page_bits_,                      // page_bits
                                                batch_indices_ptr,               // batch_indices
                                                extend_len,                      // extend_len
                                                stream);                         // stream
    } else {
      // k_ == kTopK1024
      llm_kernels::nvidia::topk::launch_topk1024(scores.GetPtr<float>(),          // scores
                                                 row_starts_ptr,                  // row_starts
                                                 row_ends_ptr,                    // row_ends
                                                 page_table.GetPtr<int32_t>(),    // page_table
                                                 page_indices.GetPtr<int32_t>(),  // page_indices_out
                                                 raw_indices_ptr,                 // raw_indices_out
                                                 score_stride,                    // score_stride
                                                 page_table_stride,               // page_table_stride
                                                 num_works,                       // num_works
                                                 page_bits_,                      // page_bits
                                                 batch_indices_ptr,               // batch_indices
                                                 extend_len,                      // extend_len
                                                 stream);                         // stream
    }
  }

  page_indices.shape = {num_works, k_};
  page_indices.dtype = DataType::TYPE_INT32;
  return Status();
}

// ---------------------------------------------------------------------------
// Multi-path v2 (topk_v2.cuh)
// ---------------------------------------------------------------------------
Status TopKLayer::ForwardV2(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // Inputs:  [0]=scores  [1]=seq_lens  [2]=page_table  [3]=workspace  [4]=metadata
  // Outputs: [0]=page_indices
  const auto& scores = input_tensors[0];
  const auto& seq_lens = input_tensors[1];
  const auto& page_table = input_tensors[2];
  const auto& workspace = input_tensors[3];
  const auto& metadata = input_tensors[4];
  auto& page_indices = output_tensors[0];

  const uint32_t batch_size = static_cast<uint32_t>(scores.shape[0]);
  const int64_t score_stride = static_cast<int64_t>(scores.shape[1]);
  const int64_t page_table_stride = static_cast<int64_t>(page_table.shape[1]);
  // workspace.shape[1] is workspace_stride_bytes (user must set this).
  const int64_t workspace_stride_bytes = static_cast<int64_t>(workspace.shape[1]);

  auto stream = context_->device->GetPrimaryComputeStream().Get();

  TopKWrapper<float, int32_t>::TransformV2(
      seq_lens.GetPtr<void>(), scores.GetPtr<void>(), page_table.GetPtr<void>(), page_indices.GetPtr<void>(),
      workspace.GetPtr<void>(), metadata.GetPtr<void>(), score_stride, page_table_stride, workspace_stride_bytes,
      batch_size, page_bits_, /*max_seq_len*/ score_stride, /*static_cluster_threshold=*/0, stream);

  page_indices.shape = {batch_size, k_};
  page_indices.dtype = DataType::TYPE_INT32;
  return Status();
}

}  // namespace ksana_llm
