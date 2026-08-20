/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/qwen/qwen_embedding_model.h"

#include <cstring>
#include <memory>
#include <vector>

#include "device/device_types.h"
#include "device/device_utils.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/request.h"

#ifdef ENABLE_CUDA
#  include <cuda_runtime.h>
#  include "csrc/kernels/nvidia/pooling/pooling.h"
#endif

namespace ksana_llm {

QwenEmbeddingModel::QwenEmbeddingModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                       const int rank, std::shared_ptr<Context> context,
                                       std::shared_ptr<BaseWeight> base_weight,
                                       std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
    : CommonModel(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  qwen_.GetModelRunConfig(model_run_config, model_config);
  CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);

  // Pre-allocate a pinned host memory buffer for async D2H copy of embedding results.
  // Size = max_batch_size * hidden_size * sizeof(bf16/fp16) (2 bytes per element).
  // This avoids the latency penalty of cudaMemcpyAsync to non-pinned (pageable) memory.
  const size_t max_batch = runtime_config.max_batch_size;
  const size_t hidden_size = model_config.hidden_units;
  const size_t elem_size = 2;  // bf16 or fp16 = 2 bytes
  pinned_host_buffer_size_ = max_batch * hidden_size * elem_size;
#ifdef ENABLE_CUDA
  HostAlloc(&pinned_host_buffer_, pinned_host_buffer_size_);
  KLLM_LOG_INFO << "QwenEmbeddingModel allocated pinned host buffer: " << pinned_host_buffer_size_ << " bytes";
#endif

  KLLM_LOG_INFO << "QwenEmbeddingModel initialized, hidden_units=" << model_config_.hidden_units
                << ", pooling_type=" << GetPoolingTypeString(model_config_.pooling_type)
                << ", normalize_embedding=" << (model_config_.normalize_embedding ? "true" : "false");
}

QwenEmbeddingModel::~QwenEmbeddingModel() {
#ifdef ENABLE_CUDA
  if (pinned_host_buffer_) {
    HostFree(pinned_host_buffer_);
    pinned_host_buffer_ = nullptr;
  }
#endif
}

Status QwenEmbeddingModel::CreateLayers(LayerCreationContext& creation_context,
                                        ModelCreationConfig& model_creation_config) {
  return qwen_.CreateLayers(creation_context, model_creation_config);
}

Status QwenEmbeddingModel::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
  STATUS_CHECK_RETURN(qwen_.Forward(residual_buffer, forwarding_context));
  return Status();
}

void QwenEmbeddingModel::UpdateEmbeddingResponse(std::vector<ForwardRequest>& forward_reqs, Tensor& output) {
  // Copy the normalized embedding from GPU to each forward request's PythonTensor response.
  // output shape: [batch_size, hidden_size]
  //
  // Strategy: Use pre-allocated pinned host memory (pinned_host_buffer_) as the D2H target
  // for cudaMemcpyAsync. Pinned memory enables truly asynchronous GPU→CPU transfer without
  // blocking the CPU thread. After stream synchronization, we memcpy from pinned buffer to
  // each request's PythonTensor response buffer.

  const size_t hidden_size = output.shape[1];
  const size_t elem_bytes = GetTypeSize(output.dtype);
  const size_t chunk_size = elem_bytes * hidden_size;

#ifdef ENABLE_CUDA
  // Use pinned buffer if available; otherwise fall back to direct copy.
  if (pinned_host_buffer_) {
    // Batch D2H copy: copy all active embeddings into contiguous pinned buffer
    size_t pinned_offset = 0;
    size_t req_offset = 0;
    for (ForwardRequest& req : forward_reqs) {
      if (!req.request_target || !req.response) {
        req_offset += 1;
        continue;
      }
      const auto it = req.request_target->find("layernorm");
      if (it == req.request_target->end()) {
        req_offset += 1;
        continue;
      }

      MemcpyAsync(static_cast<uint8_t*>(pinned_host_buffer_) + pinned_offset,
                  output.GetPtr<void>() + req_offset * chunk_size, chunk_size, MEMCPY_DEVICE_TO_HOST,
                  context_->device->GetPrimaryComputeStream());
      pinned_offset += chunk_size;
      req_offset += 1;
    }

    // Synchronize to ensure all D2H copies are complete
    StreamSynchronize(context_->device->GetPrimaryComputeStream());

    // Now copy from pinned buffer to each PythonTensor response (CPU→CPU, very fast)
    pinned_offset = 0;
    for (ForwardRequest& req : forward_reqs) {
      if (!req.request_target || !req.response) {
        continue;
      }
      const auto it = req.request_target->find("layernorm");
      if (it == req.request_target->end()) {
        continue;
      }

      PythonTensor& ret_tensor = (*req.response)["layernorm"];
      ret_tensor.shape = {1, hidden_size};
      ret_tensor.dtype = GetTypeString(output.dtype);
      ret_tensor.data.resize(chunk_size);
      std::memcpy(ret_tensor.data.data(), static_cast<uint8_t*>(pinned_host_buffer_) + pinned_offset, chunk_size);
      pinned_offset += chunk_size;
    }
  } else {
#endif
    // Fallback: direct D2H copy to pageable memory (original behavior)
    size_t req_offset = 0;
    for (ForwardRequest& req : forward_reqs) {
      if (!req.request_target || !req.response) {
        req_offset += 1;
        continue;
      }
      const auto it = req.request_target->find("layernorm");
      if (it == req.request_target->end()) {
        req_offset += 1;
        continue;
      }

      PythonTensor& ret_tensor = (*req.response)["layernorm"];
      ret_tensor.shape = {1, hidden_size};
      ret_tensor.dtype = GetTypeString(output.dtype);
      ret_tensor.data.resize(chunk_size);

      MemcpyAsync(ret_tensor.data.data(), output.GetPtr<void>() + req_offset * chunk_size, chunk_size,
                  MEMCPY_DEVICE_TO_HOST, context_->device->GetPrimaryComputeStream());
      req_offset += 1;
    }
#ifdef ENABLE_CUDA
  }
#endif
}

Status QwenEmbeddingModel::LmHead(ForwardingContext& forwarding_context,
                                  std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                  std::vector<ForwardRequest>& forward_reqs, RunMode run_mode) {
  // For embedding models, LmHead performs:
  //   1. Final RMSNorm
  //   2. Pooling (LAST/CLS via AssembleTokensHidden, or MEAN via MeanPooling kernel)
  //   3. Optional L2 normalize (controlled by model_config_.normalize_embedding)
  //   4. Copy result to PythonTensor response

  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);

  // 1. Final RMSNorm (only for pre-norm models, which Qwen3 is)
  if (lm_head_prenorm_) {
    lm_head_prenorm_->Forward(residual_buffer[0], residual_buffer[0]);
  }

  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  if (!is_multi_token_forward) {
    // For embedding, we expect multi-token forward (prefill mode).
    // In single-token mode (decode), just skip.
    KLLM_LOG_WARNING << "QwenEmbeddingModel::LmHead called in non-multi-token mode, skipping embedding output";
    return Status();
  }

  // 2. Pooling: branch based on pooling_type
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);

#ifdef ENABLE_CUDA
  const PoolingType pooling_type = model_config_.pooling_type;

  if (pooling_type == PoolingType::MEAN) {
    // MEAN pooling: average all token hidden states for each sequence.
    // Uses input_offset_uint64_tensor [batch+1] to determine each sequence's token range.
    Tensor& input_tensor = residual_buffer[0];
    Tensor& offset_tensor = forwarding_context.GetModelInput()->input_offset_uint64_tensor;
    Tensor& output_tensor = hidden_buffer_tensors_0[0];

    const int32_t batch_size = static_cast<int32_t>(offset_tensor.shape[0]) - 1;
    const int32_t hidden_size = static_cast<int32_t>(model_config_.hidden_units);
    cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

    // Ensure output tensor shape is correct: [batch_size, hidden_size]
    output_tensor.shape = {static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)};
    output_tensor.dtype = input_tensor.dtype;

    switch (input_tensor.dtype) {
      case TYPE_BF16:
        llm_kernels::nvidia::InvokeMeanPooling(input_tensor.GetPtr<__nv_bfloat16>(), offset_tensor.GetPtr<size_t>(),
                                               batch_size, hidden_size, output_tensor.GetPtr<__nv_bfloat16>(), stream);
        break;
      case TYPE_FP16:
        llm_kernels::nvidia::InvokeMeanPooling(input_tensor.GetPtr<half>(), offset_tensor.GetPtr<size_t>(), batch_size,
                                               hidden_size, output_tensor.GetPtr<half>(), stream);
        break;
      default:
        KLLM_LOG_ERROR << "Unsupported dtype for MeanPooling: " << static_cast<int>(input_tensor.dtype);
        return Status(RetCode::RET_INVALID_ARGUMENT,
                      fmt::format("Unsupported dtype for MeanPooling: {}", static_cast<int>(input_tensor.dtype)));
    }
  } else {
    // LAST / CLS pooling: extract single token via AssembleTokensHidden.
    // The token index (last or first) is controlled by Python layer's slice_pos → logits_idx.
    STATUS_CHECK_RETURN(static_cast<AssembleTokensHiddenLayer*>(assemble_tokens_hidden_layer_.get())
                            ->Forward(&residual_buffer[0],
                                      &forwarding_context.GetModelInput()->logits_idx_uint64_tensor,
                                      &hidden_buffer_tensors_0[0]));
  }
#else
  KLLM_LOG_ERROR << "QwenEmbeddingModel::LmHead requires CUDA support";
  return Status(RetCode::RET_RUNTIME_FAILED, "QwenEmbeddingModel requires CUDA support");
#endif

  // hidden_buffer_tensors_0[0] now has shape [num_seqs, hidden_size]

  // 3. Optional L2 normalize in-place
#ifdef ENABLE_CUDA
  if (model_config_.normalize_embedding) {
    Tensor& pooled = hidden_buffer_tensors_0[0];
    const int32_t batch_size = static_cast<int32_t>(pooled.shape[0]);
    const int32_t hidden_size = static_cast<int32_t>(pooled.shape[1]);
    cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

    switch (pooled.dtype) {
      case TYPE_BF16:
        llm_kernels::nvidia::InvokeL2Normalize(pooled.GetPtr<__nv_bfloat16>(), batch_size, hidden_size, stream);
        break;
      case TYPE_FP16:
        llm_kernels::nvidia::InvokeL2Normalize(pooled.GetPtr<half>(), batch_size, hidden_size, stream);
        break;
      default:
        KLLM_LOG_ERROR << "Unsupported dtype for L2 normalize: " << static_cast<int>(pooled.dtype);
        return Status(RetCode::RET_INVALID_ARGUMENT,
                      fmt::format("Unsupported dtype for L2 normalize: {}", static_cast<int>(pooled.dtype)));
    }
  }
#endif

  // 4. Copy result to response PythonTensor (includes StreamSynchronize for pinned path)
  if (rank_ == 0) {
    UpdateEmbeddingResponse(forward_reqs, hidden_buffer_tensors_0[0]);
  }

  // Synchronize the compute stream for:
  //   - rank == 0 without pinned buffer: UpdateEmbeddingResponse used async D2H copy
  //     to pageable memory, which needs an explicit sync before the host reads the data.
  //   - rank != 0: although no D2H copy happens, the preceding kernels (RMSNorm,
  //     Pooling, L2Normalize) are still running on the compute stream. We must sync
  //     to ensure they complete before the GPU buffers are reused by the next batch.
  // The pinned path on rank 0 already synchronizes inside UpdateEmbeddingResponse.
#ifdef ENABLE_CUDA
  if (!pinned_host_buffer_ || (rank_ != 0)) {
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }
#endif
  input_refit_layer_->Clear();

  return Status();
}

}  // namespace ksana_llm
