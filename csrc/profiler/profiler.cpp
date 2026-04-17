/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <unistd.h>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>


#include "oneapi/tbb/concurrent_hash_map.h"
#include "profiler/profiler.h"
#include "utils/string_utils.h"

namespace ksana_llm {
NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);

// Private constructor (friend of Singleton) initializes attribute view which lacks default ctor
struct Profiler::Impl {
  std::mutex metrics_mutex;
  std::mutex counters_mutex;
  tbb::concurrent_hash_map<std::string,
                           opentelemetry::v1::nostd::unique_ptr<opentelemetry::v1::metrics::Histogram<double>>>
      metrics;
  tbb::concurrent_hash_map<std::string,
                           opentelemetry::v1::nostd::unique_ptr<opentelemetry::v1::metrics::Counter<uint64_t>>>
      counters;
};

Profiler::Profiler()
    : resource_attributes_(),
      data_attributes_(),
      attributes_view_(data_attributes_),
      meter_(nullptr),
      impl_(std::make_unique<Impl>()) {}

Profiler::~Profiler() {
  if (impl_) {
    impl_->metrics.clear();
    impl_->counters.clear();
  }
  meter_ = nullptr;
  // Releasing provider_ triggers PeriodicReader shutdown + final flush.
  provider_ = nullptr;
  is_initialized_ = false;
}

void Profiler::InitMetrics(const ProfilerConfig& profiler_config) {
  metrics_export_url_ = profiler_config.metrics_export_url;

  for (const auto& kv : profiler_config.resource_attributes) {
    resource_attributes_[kv.first] = kv.second;
    data_attributes_[kv.first] = kv.second;
  }

  // Inject process identity so OTLP Collector treats each process as a distinct resource.
  if (!profiler_config.process_role.empty()) {
    resource_attributes_["process.role"] = profiler_config.process_role;
    data_attributes_["process.role"] = profiler_config.process_role;
  }
  if (profiler_config.executor_id >= 0) {
    auto id_str = std::to_string(profiler_config.executor_id);
    resource_attributes_["executor.id"] = id_str;
    data_attributes_["executor.id"] = id_str;
  }
  auto pid_str = std::to_string(getpid());
  resource_attributes_["process.pid"] = pid_str;
  data_attributes_["process.pid"] = pid_str;

  attributes_view_ =
      opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>>(data_attributes_);
  export_interval_millis_ = profiler_config.export_interval_millis;
  export_timeout_millis_ = profiler_config.export_timeout_millis;

  std::unique_ptr<opentelemetry::sdk::metrics::PushMetricExporter> exporter;
  if (metrics_export_url_.substr(0, 4) == "http") {
    opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions exporter_options;
    exporter_options.url = metrics_export_url_;
    exporter_options.aggregation_temporality = opentelemetry::exporter::otlp::PreferredAggregationTemporality::kDelta;
    exporter_options.content_type = opentelemetry::exporter::otlp::HttpRequestContentType::kJson;
    exporter = opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(exporter_options);
  } else if (metrics_export_url_ == "debug" || metrics_export_url_ == "DEBUG") {
    exporter = opentelemetry::exporter::metrics::OStreamMetricExporterFactory::Create(std::cout);
  } else {
    exporter = opentelemetry::exporter::metrics::OStreamMetricExporterFactory::Create(null_stream);
  }

  opentelemetry::sdk::metrics::PeriodicExportingMetricReaderOptions reader_options;
  reader_options.export_interval_millis = std::chrono::milliseconds(export_interval_millis_);
  reader_options.export_timeout_millis = std::chrono::milliseconds(export_timeout_millis_);

  auto reader =
      opentelemetry::sdk::metrics::PeriodicExportingMetricReaderFactory::Create(std::move(exporter), reader_options);
  auto context = opentelemetry::sdk::metrics::MeterContextFactory::Create(
      opentelemetry::sdk::metrics::ViewRegistryFactory::Create(),
      opentelemetry::sdk::resource::Resource::Create(resource_attributes_));
  context->AddMetricReader(std::move(reader));

  // Own the MeterProvider locally — no global Provider::SetMeterProvider().
  provider_ = std::shared_ptr<opentelemetry::metrics::MeterProvider>(
      opentelemetry::sdk::metrics::MeterProviderFactory::Create(std::move(context)));
  meter_ = provider_->GetMeter("ksana_inference_metrics", SCOPE_VERSION);
  is_initialized_ = true;
}

void Profiler::ReportMetric(const std::string& name, double value) {
  if (!is_initialized_) return;

  using MetricsMap =
      tbb::concurrent_hash_map<std::string,
                               opentelemetry::v1::nostd::unique_ptr<opentelemetry::v1::metrics::Histogram<double>>>;

  MetricsMap::accessor acc;
  bool inserted = impl_->metrics.insert(acc, name);

  if (inserted || !acc->second) {
    std::lock_guard<std::mutex> lock(impl_->metrics_mutex);
    if (!acc->second) {  // double check
      acc->second = std::move(meter_->CreateDoubleHistogram(opentelemetry::nostd::string_view(name)));
    }
  }

  acc->second->Record(value, attributes_view_, opentelemetry::context::Context{});
}

void Profiler::ReportCounter(const std::string& name, int64_t value) {
  if (!is_initialized_) return;

  using CountersMap =
      tbb::concurrent_hash_map<std::string,
                               opentelemetry::v1::nostd::unique_ptr<opentelemetry::v1::metrics::Counter<uint64_t>>>;

  CountersMap::accessor acc;
  bool inserted = impl_->counters.insert(acc, name);

  if (inserted || !acc->second) {
    std::lock_guard<std::mutex> lock(impl_->counters_mutex);
    if (!acc->second) {  // double check
      acc->second = std::move(meter_->CreateUInt64Counter(opentelemetry::nostd::string_view(name)));
    }
  }

  acc->second->Add(value, attributes_view_, opentelemetry::context::Context{});
}

namespace {
std::unique_ptr<Profiler> g_process_profiler;
std::once_flag g_profiler_init_flag;
}  // namespace

void InitProcessProfiler(const ProfilerConfig& config) {
  std::call_once(g_profiler_init_flag, [&config]() {
    g_process_profiler = std::make_unique<Profiler>();
    g_process_profiler->InitMetrics(config);
  });
}

void DestroyProcessProfiler() { g_process_profiler.reset(); }

Profiler* GetProcessProfiler() { return g_process_profiler.get(); }

}  // namespace ksana_llm
