/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/context.h"
#include "device/device_types.h"
#include "models/base/base_weight.h"
#include "models/base/buffer_manager.h"
#include "profiler/profile_event.h"
#include "runtime/forward_request.h"
#include "runtime/hidden_unit_buffer.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

enum class RunMode {
  kMain,   // default
  kNextN,  // run MTP next n
};

class BaseModel {
 public:
  // Disable a default constructor
  BaseModel();

  virtual ~BaseModel();

  // Forward model.
  // model_input_slot: ForwardingContext 内多 ModelInput 槽位选择（慢路径默认 0）。
  // prev_ring_dev: build-ahead 启用时为上一步 (req_id, token) ring slot 起始指针；非 nullptr 触发
  //   ReplaceLastTokenKernel（按 req_id 回填 placeholder）。慢路径或未启用时为 nullptr，跳过 ReplaceKernel。
  virtual Status Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs,
                         HiddenUnitDeviceBuffer* hidden_unit, RunMode run_mode = RunMode::kMain,
                         size_t model_input_slot = 0, const int64_t* prev_ring_dev = nullptr,
                         bool gpu_input_already_prepared = false) = 0;

  // build-ahead: Preprocess 线程在 slot 对应 H2D stream 上构建 ModelInput（默认空实现）。
  virtual Status PrepareModelInputGpu(std::vector<ForwardRequest>& forward_reqs, RunMode run_mode,
                                      size_t model_input_slot);

  // 取本步 slot 对应 ModelInput 的 (req_id, last_offset) 配对 tensor device 指针 + batch_size。
  // sampling 完成后供 Sampler::WriteRing 固化到 ring。默认空实现；CommonModel 转发到 GetModelInput(slot)。
  virtual void GetCurBatchPairs(size_t /*model_input_slot*/, const int64_t** out_pairs_dev, int* out_batch_size) {
    if (out_pairs_dev) *out_pairs_dev = nullptr;
    if (out_batch_size) *out_batch_size = 0;
  }

  // The output logits pointer on device, used by sampler to avoid memory copy.
  virtual float* GetLogitsPtr() = 0;

  // Release the per-request state held by the model's optional forward-meta modules (currently
  // CompressorMeta for DSv4). Default is a no-op for models that do not maintain such state.
  // Called by the scheduler when a request finishes or is recomputed.
  virtual void EvictFinishedRequests(const std::vector<int64_t>& finished_req_ids) {}

  BufferManager* GetBufferManager() { return &buffer_mgr_; }

 protected:
  std::shared_ptr<Context> context_{nullptr};

  int rank_{0};

  BufferManager buffer_mgr_;

 protected:
  // Log total buffer tensors memory used
  const size_t GetBufferTensorsMemoryUsed() { return buffer_mgr_.GetBufferTensorsMemoryUsed(); }
};

}  // namespace ksana_llm
