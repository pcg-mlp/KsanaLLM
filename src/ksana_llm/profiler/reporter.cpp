/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/profiler/reporter.h"

namespace ksana_llm {

TimeReporter::TimeReporter(const std::string &name, TimeUnit time_unit) {
  name_ = name;
  time_unit_ = time_unit;
  start_ = GetCurrentByUnit(time_unit_);
}

TimeReporter::~TimeReporter() {}

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
      KLLM_CHECK_WITH_INFO(false, "time unit is not supported.");
      break;
  }
  return 0;
}

void ReporterMetric(const std::string &name, const size_t value,
                    const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx) {
  auto context = opentelemetry::context::Context{};
  if (req_ctx && !req_ctx->empty()) {
    opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes(*req_ctx);
    Singleton<Profiler>::GetInstance()
        ->GetMeter("ksana_inference_metrics")
        ->CreateUInt64Histogram(name)
        ->Record(value, attributes, context);
  } else {
    Singleton<Profiler>::GetInstance()
        ->GetMeter("ksana_inference_metrics")
        ->CreateUInt64Histogram(name)
        ->Record(value, context);
  }
}

void ReporterMetric(const std::string &name, const float value,
                    const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx) {
  ReporterMetric(name, static_cast<double>(value), req_ctx);
}

void ReporterMetric(const std::string &name, const double value,
                    const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx) {
  auto context = opentelemetry::context::Context{};
  if (req_ctx && !req_ctx->empty()) {
    opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes(*req_ctx);
    Singleton<Profiler>::GetInstance()
        ->GetMeter("ksana_inference_metrics")
        ->CreateDoubleHistogram(name)
        ->Record(value, attributes, context);
  } else {
    Singleton<Profiler>::GetInstance()
        ->GetMeter("ksana_inference_metrics")
        ->CreateDoubleHistogram(name)
        ->Record(value, context);
  }
}

void ReportCounter(const std::string &name, const size_t value,
                   const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx) {
  if (req_ctx && !req_ctx->empty()) {
    opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes(*req_ctx);
    Singleton<Profiler>::GetInstance()
        ->GetMeter("ksana_inference_metrics")
        ->CreateUInt64Counter(name)
        ->Add(value, attributes);
  } else {
    Singleton<Profiler>::GetInstance()->GetMeter("ksana_inference_metrics")->CreateUInt64Counter(name)->Add(value);
  }
}

void ReportCounter(const std::string &name, const double value,
                   const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx) {
  if (req_ctx && !req_ctx->empty()) {
    opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes(*req_ctx);
    Singleton<Profiler>::GetInstance()
        ->GetMeter("ksana_inference_metrics")
        ->CreateDoubleCounter(name)
        ->Add(value, attributes);

  } else {
    Singleton<Profiler>::GetInstance()->GetMeter("ksana_inference_metrics")->CreateDoubleCounter(name)->Add(value);
  }
}

void ReportCounter(const std::string &name, const float value,
                   const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx) {
  ReportCounter(name, static_cast<double>(value), req_ctx);
}

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> ReportTrace(
    const std::string &span_name, const opentelemetry::trace::StartSpanOptions &carrier) {
  auto span = Singleton<Profiler>::GetInstance()->GetTracer("ksana_inference_tracer")->StartSpan(span_name, carrier);
  span->AddEvent(std::string("Processing_") + span_name);
  return span;
}

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> ReportTrace(
    const std::string &span_name,
    const HttpTextMapCarrier<const std::unordered_map<std::string, std::string>> carrier) {
  auto prop = opentelemetry::context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
  auto current_ctx = opentelemetry::context::RuntimeContext::GetCurrent();
  auto context = prop->Extract(carrier, current_ctx);

  opentelemetry::trace::StartSpanOptions options;
  options.parent = opentelemetry::trace::GetSpan(context)->GetContext();

  return ReportTrace(span_name, options);
}
}  // namespace ksana_llm
