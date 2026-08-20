/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "runtime/hidden_unit_buffer.h"
#include "runtime/model_input.h"
#include "runtime/model_instance.h"
#include "runtime/model_output.h"
#include "utils/status.h"

namespace ksana_llm {

class ModelExecutor {
 public:
  // prev_ring_dev (build-ahead Phase 1 previous-step sampler ring) is threaded via
  // model_input->prev_ring_dev, read inside Forward and passed down to CommonModel::Forward for the
  // device-side placeholder backfill. nullptr on the slow path.
  Status Forward(std::shared_ptr<NewModelInput> model_input, HiddenUnitDeviceBuffer* hidden_unit,
                 std::shared_ptr<NewModelOutput> model_output);

  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);
  void SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight);
  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager);

  std::shared_ptr<ModelInstance> GetModelInstance() const { return model_instance_; }
  std::shared_ptr<WeightInstanceInterface> GetModelWeight() const { return model_weight_; }

 private:
  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> model_weight_ = nullptr;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_ = nullptr;
};

}  // namespace ksana_llm
