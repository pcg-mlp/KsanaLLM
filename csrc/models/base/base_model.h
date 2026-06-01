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
  virtual Status Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs,
                         HiddenUnitDeviceBuffer* hidden_unit, RunMode run_mode = RunMode::kMain) = 0;

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
