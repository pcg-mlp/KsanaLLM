/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/emb_lookup_layer.h"

#include <algorithm>
#include <vector>

#include "device/device_utils.h"
#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

Status EmbLookupLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  size_t parameter_index = 0ul;
  if (parameter_index < parameters.size()) {
    use_emb_scale_ = std::any_cast<bool>(parameters[parameter_index++]);
  }
  if (parameter_index < parameters.size()) {
    emb_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  }
  if (parameter_index < parameters.size()) {
    pos_weight_ = std::any_cast<void*>(parameters[parameter_index++]);
  }

  // Kunlun XPU: xdnn::embedding requires int64_t indices, but framework provides int32_t.
  // Pre-allocate workspace for the int32 → int64 cast buffer on device.
  size_t max_token_num = runtime_config.max_step_token_num;
  if (max_token_num == 0) {
    max_token_num = 4096;  // fallback default
  }
  workspace_buffer_ = std::make_shared<Tensor>();
  workspace_buffer_->ReallocateMemory(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT64, {max_token_num});
  return Status();
}

Status EmbLookupLayer::Forward(const Tensor* input_ids, const Tensor* ids_offsets, const Tensor* prefix_offsets,
                               const Tensor* emb_weight, Tensor* output, const Tensor* steps,
                               const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_ids, ids_offsets, prefix_offsets, emb_weight, output, steps);
}

template <typename T>
Status EmbLookupLayer::ForwardT(const Tensor* input_ids, const Tensor* ids_offsets, const Tensor* prefix_offsets,
                                const Tensor* emb_weight, Tensor* output, const Tensor* steps) {
  // input_ids [token_num], TYPE_INT32
  // ids_offsets [batch_size + 1]
  // prefix_offsets [batch_size + 1]
  // emb_weight [vocab_size, hidden_units]
  // steps [token_num] (optional)
  // output: emb_output [token_num, hidden_units]
  namespace xpukernel = xpytorch::xpu::api;
  auto* xpuctx = static_cast<xpukernel::Context*>(
      context_->device->ext->GetXpuKernelContext(context_->device->GetPrimaryComputeStream().Get()));

  int64_t token_num = static_cast<int64_t>(input_ids->shape[0]);
  int64_t vocab_size = static_cast<int64_t>(emb_weight->shape[0]);
  int64_t hidden_units = static_cast<int64_t>(emb_weight->shape[1]);

  KLLM_LOG_DEBUG << "EmbLookupLayer: token_num=" << token_num << " vocab_size=" << vocab_size
                 << " hidden_units=" << hidden_units;

  const T* weight = emb_weight->GetPtr<T>();
  T* output_ptr = output->GetPtr<T>();

  // 框架的 input_ids 是 INT32，但 xdnn::embedding 要求 int64_t index。
  // 需要在 device 上做 int32 → int64 转换。
  // 使用 workspace_buffer_ 作为临时存储。
  const int32_t* input_ids_i32 = input_ids->GetPtr<int32_t>();
  int64_t* input_ids_i64 = workspace_buffer_->GetPtr<int64_t>();

  // xdnn::cast: int32 → int64
  int ret = XpuCast<int32_t, int64_t>(xpuctx, input_ids_i32, input_ids_i64, token_num);
  if (ret != 0) {
    KLLM_LOG_ERROR << "EmbLookupLayer: int32->int64 cast failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuCast int32->int64 failed in EmbLookupLayer.");
  }

  // Diagnostic: sync + read back token IDs to check for out-of-range values.
  //
  // NOTE(perf): this block is compiled out by default. It was found running
  // unconditionally on every decode step and dominated the D2H count in the
  // profiler (2 of 4 D2H per step, +4 host StreamSynchronize per step, ~50-150
  // us on the critical path — see optimization_roadmap.md P0*). Enable only
  // when hunting an EmbLookup correctness bug.
#ifdef KLLM_ENABLE_EMBLOOKUP_DIAGNOSTIC
  {
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    std::vector<int32_t> host_ids(token_num);
    MemcpyAsync(host_ids.data(), input_ids_i32, token_num * sizeof(int32_t), MEMCPY_DEVICE_TO_HOST,
                context_->device->GetPrimaryComputeStream());
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    int32_t max_id = *std::max_element(host_ids.begin(), host_ids.end());
    int32_t min_id = *std::min_element(host_ids.begin(), host_ids.end());
    KLLM_LOG_INFO << "EmbLookupLayer: input_ids min=" << min_id << " max=" << max_id << " first_5=[";
    for (int i = 0; i < std::min(static_cast<int64_t>(5), token_num); ++i) {
      KLLM_LOG_INFO << "  id[" << i << "]=" << host_ids[i];
    }
    if (max_id >= vocab_size || min_id < 0) {
      KLLM_LOG_ERROR << "EmbLookupLayer: OUT OF RANGE! max_id=" << max_id << " min_id=" << min_id
                     << " vocab_size=" << vocab_size;
    }
  }
#endif

  ret = XpuEmbeddingLookup<T>(xpuctx, input_ids_i64, weight, output_ptr, token_num, hidden_units, vocab_size);
  if (ret != 0) {
    KLLM_LOG_ERROR << "EmbLookupLayer: XpuEmbeddingLookup failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuEmbeddingLookup failed in EmbLookupLayer.");
  }

  // Dump first 8 values of embedding output for precision comparison. Same
  // rationale as above — off by default. See optimization_roadmap.md P0*.
#ifdef KLLM_ENABLE_EMBLOOKUP_DIAGNOSTIC
  {
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    std::vector<int16_t> host_out(8);
    MemcpyAsync(host_out.data(), output_ptr, 8 * sizeof(int16_t), MEMCPY_DEVICE_TO_HOST,
                context_->device->GetPrimaryComputeStream());
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    std::string s;
    for (int i = 0; i < 8; ++i) s += std::to_string(static_cast<uint16_t>(host_out[i])) + " ";
    KLLM_LOG_INFO << "EmbLookupLayer OUTPUT[0,:8] bits: " << s;
  }
#endif

  output->shape = {static_cast<size_t>(token_num), static_cast<size_t>(hidden_units)};
  output->dtype = emb_weight->dtype;
  return Status();
}

}  // namespace ksana_llm
