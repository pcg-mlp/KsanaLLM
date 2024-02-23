/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

namespace ksana_llm {

// Report time.
class TimeReporter {
 public:
  TimeReporter();

  ~TimeReporter();
};

// Report metric.
class MetricReporter {
 public:
  MetricReporter();
};

// Report event.
class EventReporter {
 public:
  EventReporter();
};

#define REPORT_TIME_MS(var)
#define REPORT_TIME_US(var)
#define REPORT_TIME_NS(var)

#define REPORT_METRIC_INT(metric1, int_value)
#define REPORT_METRIC_FLOAT(metric1, int_value)

#define REPORT_EVENT_STR(event1, str_anything)

} // namespace ksana_llm
