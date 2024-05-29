/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_manager/batch_scheduler/strategy/base_strategy.h"

namespace ksana_llm {

// The auto-batching scheduler implementation.
class AutoBatchingStrategy : public BaseScheduleStrategy {
  public:
    AutoBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config, int tp_num,
                         std::shared_ptr<BatchState> batch_state);

    virtual void Schedule() override;


  private:
    // return true if all reqs in running batch is finished.
    bool CheckBatchFinished();

    // return true if all reqs in running batch is timeout.
    bool CheckBatchTimeout();

    // Padding requests in one batch.
    void PaddingRequests();

    // Padding requests in one batch.
    void AdjustInferStage();

    // Process finished reqs.
    void FinishBatchRequests(Status finish_status);

    // Fetch new request batch from waiting queue.
    void FetchBatchRequests();
};


}  // namespace ksana_llm
