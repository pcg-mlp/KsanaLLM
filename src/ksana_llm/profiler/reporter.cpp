/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/profiler/collector.h"
#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

TimeReporter::TimeReporter(const std::string &name, TimeUnit time_unit) {
  name_ = name;
  time_unit_ = time_unit;
  start_ = GetCurrentByUnit(time_unit_);
}

TimeReporter::~TimeReporter() { GetProfileCollector()->ReportTime(name_, GetCurrentByUnit(time_unit_) - start_); }

inline time_t TimeReporter::GetCurrentByUnit(TimeUnit time_unit) {
  switch (time_unit) {
    case TimeUnit::TIME_MS:
      return ProfileTimer::GetCurrentTimeInMs();
      break;
    case TimeUnit::TIME_US:
      return ProfileTimer::GetCurrentTimeInUs();
      break;
    case TimeUnit::TIME_NS:
      return ProfileTimer::GetCurrentTimeInNs();
    default:
      NLLM_CHECK_WITH_INFO(false, "time unit is not supported.");
      break;
  }
  return 0;
}

MetricReporter::MetricReporter(const std::string &name, int64_t value) {
  GetProfileCollector()->ReportMetric(name, value);
}

MetricReporter::MetricReporter(const std::string &name, int value) {
  GetProfileCollector()->ReportMetric(name, static_cast<int64_t>(value));
}

MetricReporter::MetricReporter(const std::string &name, size_t value) {
  GetProfileCollector()->ReportMetric(name, static_cast<int64_t>(value));
}

MetricReporter::MetricReporter(const std::string &name, float value) {
  GetProfileCollector()->ReportMetric(name, value);
}

MetricReporter::MetricReporter(const std::string &name, double value) {
  GetProfileCollector()->ReportMetric(name, static_cast<float>(value));
}

EventReporter::EventReporter(const std::string &name, const std::string &message) {
  GetProfileCollector()->ReportEvent(name, message);
}

}  // namespace ksana_llm
