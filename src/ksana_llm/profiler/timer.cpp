/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/profiler/timer.h"

namespace ksana_llm {

std::time_t ProfileTimer::GetCurrentTime() {
  std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now().time_since_epoch();
  std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>(d);
  return sec.count();
}

std::time_t ProfileTimer::GetCurrentTimeInMs() {
  std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now().time_since_epoch();
  std::chrono::milliseconds milli_sec = std::chrono::duration_cast<std::chrono::milliseconds>(d);
  return milli_sec.count();
}

std::time_t ProfileTimer::GetCurrentTimeInUs() {
  std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now().time_since_epoch();
  std::chrono::microseconds micro_sec = std::chrono::duration_cast<std::chrono::microseconds>(d);
  return micro_sec.count();
}

std::time_t ProfileTimer::GetCurrentTimeInNs() {
  std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now().time_since_epoch();
  std::chrono::nanoseconds nano_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(d);
  return nano_sec.count();
}

}  // namespace ksana_llm
