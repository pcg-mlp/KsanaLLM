/* Copyright 2025 Tencent Inc.
==============================================================================*/
#pragma once

#include <memory>
#include "runtime/schedule_output.h"

namespace ksana_llm {

void MergeScheduleOutputGroupRunningRequests(std::shared_ptr<ScheduleOutputGroup>& schedule_output_group,
                                             ScheduleOutput& merged_schedule_output);

void MergeScheduleInfoForWorkers(const std::vector<ScheduleOutput*>& outputs, ScheduleOutput& merged_schedule_output);

}  // namespace ksana_llm
