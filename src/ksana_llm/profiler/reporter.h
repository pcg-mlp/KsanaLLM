/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <string>

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
struct MetricReporter {
    MetricReporter(const std::string &name, int value);
    MetricReporter(const std::string &name, size_t value);
    MetricReporter(const std::string &name, int64_t value);

    MetricReporter(const std::string &name, float value);
    MetricReporter(const std::string &name, double value);
};

// Report event.
struct EventReporter {
    EventReporter(const std::string &name, const std::string &message);
};

#define REPORT_TIME_MS(name) TimeReporter time_reporter_##name(#name, TimeReporter::TimeUnit::TIME_MS)

#define REPORT_TIME_US(name) TimeReporter time_reporter_##name(#name, TimeReporter::TimeUnit::TIME_US)

#define REPORT_TIME_NS(name) TimeReporter time_reporter_##name(#name, TimeReporter::TimeUnit::TIME_NS)

#define REPORT_METRIC(name, value) MetricReporter metirc_reporter_##name(#name, value)

#define REPORT_EVENT(name, message) EventReporter event_reporter_##name(#name, #message)

}  // namespace ksana_llm
