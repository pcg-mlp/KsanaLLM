/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "numerous_llm/batch_manager/batch_scheduler/batch_scheduler.h"
#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class BatchManager {
public:
  BatchManager(const BatchManagerConfig &batch_manager_config);
  ~BatchManager();

  // Register a model instance to current batch manager.
  Status RegisterModelInstance(const ModelInstance &model_instance);

  // Enqueue a request to waiting queue.
  Status Enqueue(int req_id, const std::vector<TensorMap> &tensor_maps,
                 const std::vector<SamplingConfig> &sampling_configs);

  // Wait request done and return output tensor maps.
  Status WaitDone(int req_id, std::vector<TensorMap> &tensor_maps);

  // Wait all requests done.
  Status WaitAllDone();

private:
  // The batch scheduler.
  std::shared_ptr<BatchScheduler> batch_scheduler_ = nullptr;

  // Used to cache kv cache of prefix part of input.
  std::shared_ptr<ContextCache> context_cache_ = nullptr;
};

} // namespace numerous_llm
