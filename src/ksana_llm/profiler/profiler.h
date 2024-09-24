/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <streambuf>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/environment.h"

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/exporters/ostream/metric_exporter_factory.h"
#include "opentelemetry/exporters/ostream/span_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_options.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h"
#include "opentelemetry/ext/http/client/http_client.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/sdk/common/global_log_handler.h"
#include "opentelemetry/sdk/metrics/aggregation/default_aggregation.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_factory.h"
#include "opentelemetry/sdk/metrics/meter.h"
#include "opentelemetry/sdk/metrics/meter_context_factory.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/metrics/meter_provider_factory.h"
#include "opentelemetry/sdk/metrics/view/view_registry_factory.h"
#include "opentelemetry/sdk/trace/exporter.h"
#include "opentelemetry/sdk/trace/processor.h"
#include "opentelemetry/sdk/trace/simple_processor_factory.h"
#include "opentelemetry/sdk/trace/tracer_context.h"
#include "opentelemetry/sdk/trace/tracer_context_factory.h"
#include "opentelemetry/sdk/trace/tracer_provider_factory.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"
#include "opentelemetry/trace/provider.h"

namespace ksana_llm {
class NullBuffer : public std::streambuf {
  virtual int overflow(int c) { return c; }
};
template <typename T>
class HttpTextMapCarrier : public opentelemetry::context::propagation::TextMapCarrier {
 public:
  explicit HttpTextMapCarrier(T& headers) : headers_(headers) {}
  HttpTextMapCarrier() = default;
  virtual opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view key) const noexcept override {
    auto it = headers_.find(std::string(key));
    if (it != headers_.end()) {
      return it->second;
    }
    return "";
  }

  virtual void Set(opentelemetry::nostd::string_view key, opentelemetry::nostd::string_view value) noexcept override {}

  T headers_;
};

// Use to collect profile data from different modules, must be thread-safe.
class Profiler {
 public:
  Profiler() {}
  void Init(const ProfilerConfig& profiler_config);
  ~Profiler();

  void InitTracer();

  void CleanupTracer();

  opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> GetTracer(std::string tracer_name);

  void InitMetrics();

  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> GetMeter(std::string meter_name);

  void CleanupMetrics();

 private:
  std::string trace_export_url_;
  std::string metrics_export_url_;
  uint64_t export_interval_millis_;
  uint64_t export_timeout_millis_;
  opentelemetry::sdk::common::AttributeMap attr_;
};
}  // namespace ksana_llm
