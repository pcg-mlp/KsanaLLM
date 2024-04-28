/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/batch_scheduler/strategy/strategy_factory.h"

#include "ksana_llm/batch_manager/batch_scheduler/strategy/auto_batching.h"
#include "ksana_llm/batch_manager/batch_scheduler/strategy/continuous_batching.h"

namespace ksana_llm {

std::shared_ptr<BaseScheduleStrategy> ScheduleStrategyFactory::CreateScheduleStrategy(
  const BatchSchedulerConfig &batch_scheduler_config, std::shared_ptr<Context> context,
  std::shared_ptr<BatchState> batch_state) {
  if (batch_scheduler_config.schedule_strategy == ScheduleStrategy::CONTINUOUS_BATCHING) {
    NLLM_LOG_DEBUG << "Continuous-batching scheduler created.";
    return std::make_shared<ContinuousBatchingStrategy>(batch_scheduler_config, context, batch_state);
  } else if (batch_scheduler_config.schedule_strategy == ScheduleStrategy::AUTO_BATCHING) {
    NLLM_LOG_DEBUG << "Auto-batching scheduler created.";
    return std::make_shared<AutoBatchingStrategy>(batch_scheduler_config, context, batch_state);
  }
  return nullptr;
}

}  // namespace ksana_llm
