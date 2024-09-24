/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <map>
#include <string>
#include "ksana_llm/profiler/profiler.h"
#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

// Report time.
struct TimeReporter {
  // The time unit.
  enum TimeUnit { TIME_MS, TIME_US, TIME_NS };

  TimeReporter(const std::string &name, TimeUnit time_unit);

  ~TimeReporter();

 private:
  inline time_t GetCurrentByUnit(TimeUnit time_unit);

 private:
  // The profile name.
  std::string name_;

  TimeUnit time_unit_;
  time_t start_;
};

// Report metric.
void ReporterMetric(const std::string &name, const size_t value,
                    const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx = nullptr);
void ReporterMetric(const std::string &name, const float value,
                    const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx = nullptr);
void ReporterMetric(const std::string &name, const double value,
                    const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx = nullptr);

// Report counter
void ReportCounter(const std::string &name, const size_t value,
                   const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx = nullptr);
void ReportCounter(const std::string &name, const double value,
                   const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx = nullptr);
void ReportCounter(const std::string &name, const float value,
                   const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx = nullptr);

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> ReportTrace(
    const std::string &span_name, const opentelemetry::trace::StartSpanOptions &carrier);

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> ReportTrace(
    const std::string &span_name, const HttpTextMapCarrier<const std::unordered_map<std::string, std::string>> carrier);

#define REPORT_TIME_MS(name) TimeReporter time_reporter_##name(#name, TimeReporter::TimeUnit::TIME_MS)

#define REPORT_TIME_US(name) TimeReporter time_reporter_##name(#name, TimeReporter::TimeUnit::TIME_US)

#define REPORT_TIME_NS(name) TimeReporter time_reporter_##name(#name, TimeReporter::TimeUnit::TIME_NS)

#define REPORT_COUNTER(name, value, ...) ReportCounter(std::string(#name), value, ##__VA_ARGS__)

#define REPORT_METRIC(name, value, ...) ReporterMetric(std::string(#name), value, ##__VA_ARGS__)

#define REPORT_TRACE(span_name, carrier) ReportTrace(std::string(#span_name), carrier)
}  // namespace ksana_llm
