/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "ksana_llm/profiler/profiler.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {
NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);

void Profiler::Init(const ProfilerConfig& profiler_config) {
  trace_export_url_ = profiler_config.trace_export_url;
  metrics_export_url_ = profiler_config.metrics_export_url;
  for (const auto& kv : profiler_config.resource_attributes) {
    attr_[kv.first] = kv.second;
  }
  export_interval_millis_ = profiler_config.export_interval_millis;
  export_timeout_millis_ = profiler_config.export_timeout_millis;
}

Profiler::~Profiler() {
  CleanupTracer();
  CleanupMetrics();
}

void Profiler::InitTracer() {
  std::unique_ptr<opentelemetry::sdk::trace::SpanExporter> exporter;

  if (trace_export_url_ != "") {
    opentelemetry::exporter::otlp::OtlpHttpExporterOptions exporter_options;
    exporter_options.url = trace_export_url_;
    exporter = opentelemetry::exporter::otlp::OtlpHttpExporterFactory::Create(exporter_options);
  } else {
    // By default, data is export to the black hole file
    exporter = opentelemetry::exporter::trace::OStreamSpanExporterFactory::Create(null_stream);
  }

  auto processor = opentelemetry::sdk::trace::SimpleSpanProcessorFactory::Create(std::move(exporter));
  std::vector<std::unique_ptr<opentelemetry::sdk::trace::SpanProcessor>> processors;
  processors.push_back(std::move(processor));

  // Default is an always-on sampler.
  std::unique_ptr<opentelemetry::sdk::trace::TracerContext> context =
      opentelemetry::sdk::trace::TracerContextFactory::Create(std::move(processors),
                                                              opentelemetry::sdk::resource::Resource::Create(attr_));
  std::shared_ptr<opentelemetry::trace::TracerProvider> provider =
      opentelemetry::sdk::trace::TracerProviderFactory::Create(std::move(context));

  // Set the global trace provider
  opentelemetry::trace::Provider::SetTracerProvider(provider);

  // set global propagator
  opentelemetry::context::propagation::GlobalTextMapPropagator::SetGlobalPropagator(
      opentelemetry::nostd::shared_ptr<opentelemetry::context::propagation::TextMapPropagator>(
          new opentelemetry::trace::propagation::HttpTraceContext()));
}

void Profiler::CleanupTracer() {
  std::shared_ptr<opentelemetry::trace::TracerProvider> none;
  opentelemetry::trace::Provider::SetTracerProvider(none);
}

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> Profiler::GetTracer(std::string tracer_name) {
  auto provider = opentelemetry::trace::Provider::GetTracerProvider();
  return provider->GetTracer(tracer_name);
}

void Profiler::InitMetrics() {
  std::unique_ptr<opentelemetry::sdk::metrics::PushMetricExporter> exporter;
  if (metrics_export_url_ != "") {
    opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions exporter_options;
    exporter_options.url = metrics_export_url_;
    exporter_options.aggregation_temporality = opentelemetry::exporter::otlp::PreferredAggregationTemporality::kDelta;
    exporter_options.content_type = opentelemetry::exporter::otlp::HttpRequestContentType::kJson;
    exporter = opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(exporter_options);
  } else {
    // By default, data is export to the black hole file
    exporter = opentelemetry::exporter::metrics::OStreamMetricExporterFactory::Create(null_stream);
  }

  // Initialize and set the global MeterProvider
  opentelemetry::sdk::metrics::PeriodicExportingMetricReaderOptions reader_options;
  reader_options.export_interval_millis = std::chrono::milliseconds(export_interval_millis_);
  reader_options.export_timeout_millis = std::chrono::milliseconds(export_timeout_millis_);

  auto reader =
      opentelemetry::sdk::metrics::PeriodicExportingMetricReaderFactory::Create(std::move(exporter), reader_options);
  auto context = opentelemetry::sdk::metrics::MeterContextFactory::Create(
      opentelemetry::sdk::metrics::ViewRegistryFactory::Create(),
      opentelemetry::sdk::resource::Resource::Create(attr_));
  context->AddMetricReader(std::move(reader));
  auto u_provider = opentelemetry::sdk::metrics::MeterProviderFactory::Create(std::move(context));
  std::shared_ptr<opentelemetry::metrics::MeterProvider> provider(std::move(u_provider));

  opentelemetry::metrics::Provider::SetMeterProvider(provider);
}

opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> Profiler::GetMeter(std::string meter_name) {
  auto provider = opentelemetry::metrics::Provider::GetMeterProvider();
  return provider->GetMeter(meter_name, "1.2.0");
}

void Profiler::CleanupMetrics() {
  std::shared_ptr<opentelemetry::metrics::MeterProvider> none;
  opentelemetry::metrics::Provider::SetMeterProvider(none);
}

}  // namespace ksana_llm
