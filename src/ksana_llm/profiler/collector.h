/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <mutex>
#include <string>
#include <thread>

#include "ksana_llm/profiler/writer.h"

namespace ksana_llm {

// Use to collect profile data from different modules, must be thread-safe.
class ProfileCollector {
 public:
  ProfileCollector();
  ~ProfileCollector();

  // Report values of different types.
  void Report(const std::string& name, int64_t val);
  void Report(const std::string& name, float val);
  void Report(const std::string& name, const std::string& val);

 private:
  // Start the process handle
  void StartHandle();

  // Stop the process handle
  void StopHandle();

  // The process logic.
  void Process();

 private:
  std::thread process_thread_;

  // The mutex to protect data.
  std::mutex mutex_;

  // The file writer of current collector.
  ProfileWriter writer_;

  // Used to control process thread.
  bool terminated_ = false;
};

}  // namespace ksana_llm
