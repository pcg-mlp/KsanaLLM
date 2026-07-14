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
  Status Forward(std::shared_ptr<NewModelInput> model_input, HiddenUnitDeviceBuffer* hidden_unit,
                 std::shared_ptr<NewModelOutput> model_output);

  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);
  void SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight);
  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager);

 private:
  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> model_weight_ = nullptr;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_ = nullptr;
};

}  // namespace ksana_llm
