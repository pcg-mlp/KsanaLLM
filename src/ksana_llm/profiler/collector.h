/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ksana_llm/profiler/writer.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

// Use to collect profile data from different modules, must be thread-safe.
class ProfileCollector {
 public:
  // The stat result.
  template <class T>
  struct StatResult {
    size_t size = 0;
    T mean = (T)0;
    T min = (T)0;
    T max = (T)0;
  };

 public:
  ProfileCollector(const ProfilerConfig& profiler_config);
  ~ProfileCollector();

  // Report values of different types.
  void ReportTime(const std::string& name, time_t val);
  void ReportMetric(const std::string& name, float val);
  void ReportMetric(const std::string& name, int64_t val);
  void ReportEvent(const std::string& name, const std::string& val);

  // Start the process handle
  void Start();

  // Stop the process handle
  void Stop();

 private:
  // The process logic.
  void Process();

  template <typename T>
  void UpdateProfileCollect(std::unordered_map<std::string, std::vector<T>>& collect, const std::string& name, T val) {
    if (collect.find(name) == collect.end()) {
      std::vector<T> vec;
      vec.reserve(profiler_config_.stat_buffer_size);
      collect[name] = vec;
    }
    collect[name].push_back(val);
  }

  template <typename T>
  void CalcProfileCollect(std::unordered_map<std::string, std::vector<T>>& collect,
                          std::unordered_map<std::string, StatResult<T>>& stat_results) {
    for (auto& [name, vec] : collect) {
      size_t size = vec.size();
      if (size > 0) {
        StatResult<T> stat_result;
        stat_result.size = size;
        stat_result.mean = std::accumulate(vec.begin(), vec.end(), (T)0) / size;
        stat_result.min = *min_element(vec.begin(), vec.end());
        stat_result.max = *max_element(vec.begin(), vec.end());
        stat_results[name] = stat_result;
      }
    }
  }

  template <typename T>
  void SaveProfileCollect(const std::string& datetime, std::unordered_map<std::string, StatResult<T>>& stat_results,
                          std::vector<std::string>& results) {
    for (auto& [name, stat_result] : stat_results) {
      if (std::is_same<T, time_t>::value || std::is_same<T, int64_t>::value) {
        results.push_back(FormatStr("%s|%s|record_num=%d mean=%d min=%d max=%d", datetime.c_str(), name.c_str(),
                                    stat_result.size, stat_result.mean, stat_result.min, stat_result.max));
      } else if (std::is_same<T, float>::value) {
        results.push_back(FormatStr("%s|%s|record_num=%d mean=%.2f min=%.2f max=%.2f", datetime.c_str(), name.c_str(),
                                    stat_result.size, stat_result.mean, stat_result.min, stat_result.max));
      }
    }
  }

  void SaveProfileCollect(const std::string& datetime,
                          std::unordered_map<std::string, std::vector<std::string>>& event_results,
                          std::vector<std::string>& results) {
    for (auto& [name, vec] : event_results) {
      for (auto& event : vec) {
        results.push_back(FormatStr("%s|%s|%s", datetime.c_str(), name.c_str(), event.c_str()));
      }
    }
  }

 private:
  // The config of profile collector.
  ProfilerConfig profiler_config_;

  std::unique_ptr<std::thread> process_thread_ = nullptr;

  // The mutex to protect data collect.
  std::mutex time_mutex_;
  std::mutex metric_int_mutex_;
  std::mutex metric_float_mutex_;
  std::mutex event_mutex_;

  // The file writer of current collector.
  std::shared_ptr<ProfileWriter> profile_writer_ = nullptr;

  // Used to control process thread.
  bool terminated_ = false;

  // The reported buffer.
  std::unordered_map<std::string, std::vector<time_t>> time_collect_;
  std::unordered_map<std::string, std::vector<int64_t>> metric_int_collect_;
  std::unordered_map<std::string, std::vector<float>> metric_float_collect_;
  std::unordered_map<std::string, std::vector<std::string>> event_collect_;

  // The async report threadpool.
  std::shared_ptr<ThreadPool> threadpool_ = nullptr;
};

// Set a global profile collector.
void SetProfileCollector(ProfileCollector* profile_collector);

// Get the global profile collector.
ProfileCollector* GetProfileCollector();

}  // namespace ksana_llm
