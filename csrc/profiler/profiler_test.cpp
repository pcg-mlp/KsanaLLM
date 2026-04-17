/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "profiler.h"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include "reporter.h"

namespace ksana_llm {

class ProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_config.metrics_export_url = "debug";
    test_config.resource_attributes = {{"service.name", "test_service"}, {"environment", "test"}, {"version", "1.0.0"}};
    test_config.export_interval_millis = 5000;
    test_config.export_timeout_millis = 2000;
  }

  void TearDown() override { DestroyProcessProfiler(); }

  ProfilerConfig test_config;
};

TEST_F(ProfilerTest, DefaultConstructorInitialization) {
  auto new_profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(new_profiler->ReportMetric("test_metric", 1.0));
  EXPECT_NO_THROW(new_profiler->ReportCounter("test_counter", 1));
}

TEST_F(ProfilerTest, CopyMoveConstructorsDeleted) {
  static_assert(!std::is_copy_constructible_v<Profiler>);
  static_assert(!std::is_copy_assignable_v<Profiler>);
  static_assert(!std::is_move_constructible_v<Profiler>);
  static_assert(!std::is_move_assignable_v<Profiler>);
}

TEST_F(ProfilerTest, InitMetricsWithHttpExporter) {
  ProfilerConfig config = test_config;
  config.metrics_export_url = "http://localhost:4318/v1/metrics";

  auto profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(profiler->InitMetrics(config));
  EXPECT_NO_THROW(profiler->ReportMetric("http_test_metric", 42.5));
  EXPECT_NO_THROW(profiler->ReportCounter("http_test_counter", 10));
}

TEST_F(ProfilerTest, InitMetricsWithDebugExporter) {
  ProfilerConfig config = test_config;
  config.metrics_export_url = "debug";

  auto profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(profiler->InitMetrics(config));
  EXPECT_NO_THROW(profiler->ReportMetric("debug_test_metric", 123.45));
  EXPECT_NO_THROW(profiler->ReportCounter("debug_test_counter", 5));
}

TEST_F(ProfilerTest, InitMetricsWithDebugUppercaseExporter) {
  ProfilerConfig config = test_config;
  config.metrics_export_url = "DEBUG";

  auto profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(profiler->InitMetrics(config));
  EXPECT_NO_THROW(profiler->ReportMetric("debug_upper_metric", 67.89));
  EXPECT_NO_THROW(profiler->ReportCounter("debug_upper_counter", 15));
}

TEST_F(ProfilerTest, InitMetricsWithDefaultExporter) {
  ProfilerConfig config = test_config;
  config.metrics_export_url = "unknown_protocol";

  auto profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(profiler->InitMetrics(config));
  EXPECT_NO_THROW(profiler->ReportMetric("default_test_metric", 99.99));
  EXPECT_NO_THROW(profiler->ReportCounter("default_test_counter", 25));
}

TEST_F(ProfilerTest, InitMetricsWithEmptyConfig) {
  ProfilerConfig empty_config;

  auto profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(profiler->InitMetrics(empty_config));
  EXPECT_NO_THROW(profiler->ReportMetric("empty_config_metric", 1.1));
  EXPECT_NO_THROW(profiler->ReportCounter("empty_config_counter", 1));
}

TEST_F(ProfilerTest, InitMetricsWithResourceAttributes) {
  auto profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(profiler->InitMetrics(test_config));
  EXPECT_NO_THROW(profiler->ReportMetric("attr_test_metric", 50.0));
  EXPECT_NO_THROW(profiler->ReportCounter("attr_test_counter", 20));
}

TEST_F(ProfilerTest, InitMetricsWithProcessRole) {
  ProfilerConfig config = test_config;
  config.process_role = "engine";

  auto profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(profiler->InitMetrics(config));
  EXPECT_NO_THROW(profiler->ReportMetric("engine_metric", 1.0));
}

TEST_F(ProfilerTest, InitMetricsWithExecutorRole) {
  ProfilerConfig config = test_config;
  config.process_role = "executor";
  config.executor_id = 3;

  auto profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(profiler->InitMetrics(config));
  EXPECT_NO_THROW(profiler->ReportCounter("executor_counter", 42));
}

TEST_F(ProfilerTest, ReportMetricNewMetric) {
  auto profiler = std::make_unique<Profiler>();
  profiler->InitMetrics(test_config);

  EXPECT_NO_THROW(profiler->ReportMetric("new_metric_test", 123.456));
  EXPECT_NO_THROW(profiler->ReportMetric("new_metric_test", 789.012));
}

TEST_F(ProfilerTest, ReportMetricExistingMetric) {
  auto profiler = std::make_unique<Profiler>();
  profiler->InitMetrics(test_config);

  profiler->ReportMetric("existing_metric", 1.0);
  EXPECT_NO_THROW(profiler->ReportMetric("existing_metric", 2.0));
  EXPECT_NO_THROW(profiler->ReportMetric("existing_metric", 3.0));
  EXPECT_NO_THROW(profiler->ReportMetric("existing_metric", 4.0));
}

TEST_F(ProfilerTest, ReportMetricVariousValues) {
  auto profiler = std::make_unique<Profiler>();
  profiler->InitMetrics(test_config);

  EXPECT_NO_THROW(profiler->ReportMetric("zero_metric", 0.0));
  EXPECT_NO_THROW(profiler->ReportMetric("negative_metric", -123.45));
  EXPECT_NO_THROW(profiler->ReportMetric("large_metric", 1e10));
  EXPECT_NO_THROW(profiler->ReportMetric("small_metric", 1e-10));
  EXPECT_NO_THROW(profiler->ReportMetric("inf_metric", std::numeric_limits<double>::infinity()));
}

TEST_F(ProfilerTest, ReportCounterNewCounter) {
  auto profiler = std::make_unique<Profiler>();
  profiler->InitMetrics(test_config);

  EXPECT_NO_THROW(profiler->ReportCounter("new_counter_test", 100));
  EXPECT_NO_THROW(profiler->ReportCounter("new_counter_test", 200));
}

TEST_F(ProfilerTest, ReportCounterExistingCounter) {
  auto profiler = std::make_unique<Profiler>();
  profiler->InitMetrics(test_config);

  profiler->ReportCounter("existing_counter", 1);
  EXPECT_NO_THROW(profiler->ReportCounter("existing_counter", 2));
  EXPECT_NO_THROW(profiler->ReportCounter("existing_counter", 3));
  EXPECT_NO_THROW(profiler->ReportCounter("existing_counter", 4));
}

TEST_F(ProfilerTest, ReportCounterVariousValues) {
  auto profiler = std::make_unique<Profiler>();
  profiler->InitMetrics(test_config);

  EXPECT_NO_THROW(profiler->ReportCounter("zero_counter", 0));
  EXPECT_NO_THROW(profiler->ReportCounter("small_counter", 1));
  EXPECT_NO_THROW(profiler->ReportCounter("large_counter", 1000000));
  EXPECT_NO_THROW(profiler->ReportCounter("max_counter", std::numeric_limits<int64_t>::max()));
}

TEST_F(ProfilerTest, ReportMetricsWithoutInitialization) {
  auto profiler = std::make_unique<Profiler>();
  EXPECT_NO_THROW(profiler->ReportMetric("uninit_metric", 1.0));
  EXPECT_NO_THROW(profiler->ReportCounter("uninit_counter", 1));
}

TEST_F(ProfilerTest, Destructor) {
  {
    auto temp_profiler = std::make_unique<Profiler>();
    temp_profiler->InitMetrics(test_config);
  }
  SUCCEED();
}

TEST_F(ProfilerTest, ProcessProfilerRegistry) {
  EXPECT_EQ(GetProcessProfiler(), nullptr);

  InitProcessProfiler(test_config);
  auto* p1 = GetProcessProfiler();
  EXPECT_NE(p1, nullptr);

  // Second init is a no-op (std::call_once).
  InitProcessProfiler(test_config);
  EXPECT_EQ(GetProcessProfiler(), p1);

  EXPECT_NO_THROW(p1->ReportMetric("process_metric", 1.0));
  EXPECT_NO_THROW(p1->ReportCounter("process_counter", 1));

  DestroyProcessProfiler();
  EXPECT_EQ(GetProcessProfiler(), nullptr);
}

TEST_F(ProfilerTest, ReportMacrosWithoutInit) {
  // Macros should be no-ops when profiler is not initialized.
  EXPECT_NO_THROW(REPORT_METRIC("macro_metric", 1.0));
  EXPECT_NO_THROW(REPORT_COUNTER("macro_counter", 1));
}

TEST_F(ProfilerTest, ReportMacrosWithInit) {
  InitProcessProfiler(test_config);
  EXPECT_NO_THROW(REPORT_METRIC("macro_metric", 42.0));
  EXPECT_NO_THROW(REPORT_COUNTER("macro_counter", 10));
}

TEST_F(ProfilerTest, ThreadSafety) {
  auto profiler = std::make_unique<Profiler>();
  profiler->InitMetrics(test_config);

  const int num_threads = 10;
  const int reports_per_thread = 100;
  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&profiler, i, reports_per_thread]() {
      for (int j = 0; j < reports_per_thread; ++j) {
        profiler->ReportMetric("thread_metric_" + std::to_string(i), j * 1.0);
        profiler->ReportCounter("thread_counter_" + std::to_string(i), j);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  SUCCEED();
}

TEST_F(ProfilerTest, ManyMetrics) {
  auto profiler = std::make_unique<Profiler>();
  profiler->InitMetrics(test_config);

  const int num_metrics = 1000;
  for (int i = 0; i < num_metrics; ++i) {
    EXPECT_NO_THROW(profiler->ReportMetric("metric_" + std::to_string(i), i * 1.0));
    EXPECT_NO_THROW(profiler->ReportCounter("counter_" + std::to_string(i), i));
  }
}

TEST_F(ProfilerTest, MetricNameEdgeCases) {
  auto profiler = std::make_unique<Profiler>();
  profiler->InitMetrics(test_config);

  EXPECT_NO_THROW(profiler->ReportMetric("", 1.0));
  EXPECT_NO_THROW(profiler->ReportCounter("", 1));

  std::string long_name(1000, 'a');
  EXPECT_NO_THROW(profiler->ReportMetric(long_name, 1.0));
  EXPECT_NO_THROW(profiler->ReportCounter(long_name, 1));

  EXPECT_NO_THROW(profiler->ReportMetric("metric.with.dots", 1.0));
  EXPECT_NO_THROW(profiler->ReportMetric("metric_with_underscores", 1.0));
  EXPECT_NO_THROW(profiler->ReportMetric("metric-with-dashes", 1.0));
}

}  // namespace ksana_llm
