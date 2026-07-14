/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <vector>

namespace ksana_llm {

// called by child process.
void MonitorParent();

// called by parent process.
void MonitorChildren();

// only specific pids is monitored.
void SetMonitoredChildPids(const std::vector<int64_t>& pids);

}  // namespace ksana_llm
