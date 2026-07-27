/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "configure/schedule_config_parser.h"
#include "runtime/infer_request.h"
#include "utils/status.h"

namespace ksana_llm {

// Validate the request fields shared by all scheduler admission paths.
// This function has no side effects: callers remain responsible for setting
// completion state and notifying waiters according to their group semantics.
//
// NOTE: Kept as a free function because BatchScheduler and
// EventDrivenScheduleProcessor do not share a common base that owns
// BatchSchedulerConfig (EDS uses ScheduleBackendInterface instead).
Status ValidateRequestAdmission(const std::shared_ptr<InferRequest>& req, const BatchSchedulerConfig& config);

}  // namespace ksana_llm
