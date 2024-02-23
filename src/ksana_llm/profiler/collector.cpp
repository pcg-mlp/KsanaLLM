/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/profiler/collector.h"

namespace ksana_llm {

ProfileCollector::ProfileCollector() { StartHandle(); }

ProfileCollector::~ProfileCollector() { StopHandle(); }

void ProfileCollector::StartHandle() {
}

void ProfileCollector::StopHandle() {
}

void ProfileCollector::Process() {
}

void ProfileCollector::Report(const std::string& name, int64_t val) {}

void ProfileCollector::Report(const std::string& name, float val) {}

void ProfileCollector::Report(const std::string& name, const std::string& val) {}

}  // namespace ksana_llm
