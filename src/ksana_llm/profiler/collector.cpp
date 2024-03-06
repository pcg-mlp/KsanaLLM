/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "ksana_llm/profiler/collector.h"
#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

static ProfileCollector* g_profile_collector = nullptr;

void SetProfileCollector(ProfileCollector* profile_collector) { g_profile_collector = profile_collector; }

ProfileCollector* GetProfileCollector() { return g_profile_collector; }

ProfileCollector::ProfileCollector(const ProfilerConfig& profiler_config) {
  profiler_config_ = profiler_config;
  threadpool_ = std::make_shared<ThreadPool>(profiler_config_.report_threadpool_size);

  profile_writer_ = std::make_shared<ProfileWriter>();
}

ProfileCollector::~ProfileCollector() {
  if (!terminated_) {
    Stop();
  }
}

void ProfileCollector::Start() {
  process_thread_ = std::unique_ptr<std::thread>(new std::thread(&ProfileCollector::Process, this));
  threadpool_->Start();
}

void ProfileCollector::Stop() {
  terminated_ = true;
  threadpool_->Stop();
  process_thread_->join();
}

void ProfileCollector::Process() {
  time_t current_time;
  time_t last_stat_time = ProfileTimer::GetCurrentTime();
  while (!terminated_) {
    current_time = ProfileTimer::GetCurrentTime();
    if (last_stat_time + profiler_config_.stat_interval_second > current_time) {
      std::this_thread::sleep_for(
          std::chrono::seconds(last_stat_time + profiler_config_.stat_interval_second - current_time));
    }
    last_stat_time = ProfileTimer::GetCurrentTime();

    // Start to calculate reported data.
    std::string datetime = ProfileTimer::GetCurrentTimeInStr();

    std::unordered_map<std::string, StatResult<time_t>> time_stat_results;
    std::unordered_map<std::string, StatResult<int64_t>> metric_int_stat_results;
    std::unordered_map<std::string, StatResult<float>> metric_float_stat_results;
    std::unordered_map<std::string, std::vector<std::string>> event_stat_results;

    {
      std::unique_lock<std::mutex> lock(time_mutex_);
      if (!time_collect_.empty()) {
        CalcProfileCollect(time_collect_, time_stat_results);
        time_collect_.clear();
      }
    }

    {
      std::unique_lock<std::mutex> lock(metric_int_mutex_);
      if (!metric_int_collect_.empty()) {
        CalcProfileCollect(metric_int_collect_, metric_int_stat_results);
        metric_int_collect_.clear();
      }
    }

    {
      std::unique_lock<std::mutex> lock(metric_float_mutex_);
      if (!metric_float_collect_.empty()) {
        CalcProfileCollect(metric_float_collect_, metric_float_stat_results);
        metric_float_collect_.clear();
      }
    }

    {
      std::unique_lock<std::mutex> lock(event_mutex_);
      event_stat_results.swap(event_collect_);
    }

    // Write to stat file.
    std::vector<std::string> results;
    SaveProfileCollect(datetime, time_stat_results, results);
    SaveProfileCollect(datetime, metric_int_stat_results, results);
    SaveProfileCollect(datetime, metric_float_stat_results, results);
    SaveProfileCollect(datetime, event_stat_results, results);

    if (!results.empty()) {
      profile_writer_->Write(results);
    }
  }
}

void ProfileCollector::ReportTime(const std::string& name, time_t val) {
  threadpool_->Submit([=]() {
    std::unique_lock<std::mutex> lock(time_mutex_);
    UpdateProfileCollect(time_collect_, name, val);
  });
}

void ProfileCollector::ReportMetric(const std::string& name, int64_t val) {
  threadpool_->Submit([=]() {
    std::unique_lock<std::mutex> lock(metric_int_mutex_);
    UpdateProfileCollect(metric_int_collect_, name, val);
  });
}

void ProfileCollector::ReportMetric(const std::string& name, float val) {
  threadpool_->Submit([=]() {
    std::unique_lock<std::mutex> lock(metric_float_mutex_);
    UpdateProfileCollect(metric_float_collect_, name, val);
  });
}

void ProfileCollector::ReportEvent(const std::string& name, const std::string& message) {
  threadpool_->Submit([=]() {
    std::unique_lock<std::mutex> lock(event_mutex_);
    UpdateProfileCollect(event_collect_, name, message);
  });
}

}  // namespace ksana_llm
