/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <string>

namespace ksana_llm {

class ProfileTimer {
 public:
  // Get current time in seconds.
  static std::time_t GetCurrentTime();

  // Get current time in ms.
  static std::time_t GetCurrentTimeInMs();

  // Get current time in us.
  static std::time_t GetCurrentTimeInUs();

  // Get current time in ns.
  static std::time_t GetCurrentTimeInNs();

  // Get current time in string.
  static std::string GetCurrentTimeInStr(const std::string& format = "%Y-%m-%d %H:%M:%S");
};

}  // namespace ksana_llm
