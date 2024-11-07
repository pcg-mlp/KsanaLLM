/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <queue>
#include <string>
#include <thread>

#include "timer.h"

namespace ksana_llm {

// Define event types to specify display style (e.g. color)
enum TraceEventType {
  SchedBegin,       // Request is prepared to add to scheduler
  DropReq,          // Request is dropped because of queue full, timeout
  FinishReq,        // Request is success
  Prefill,          // Forward request in stage Prefilling
  Decode,           // Forward request in stage Decoding
  SwapOut,          // Request is swapped out
  SwapIn,           // Request is swapped in
  InputTokenNum,    // Metric: num of tokens in forward step which consume kv-cache
  ForwardTokenNum,  // Metric: num of tokens in forward step which consume flops
  TokenNumPerSec,   // Metric: forward token num in a second
  UsableBlockNum,   // Metric: Usable block num
  FutureBlockNum    // Metric: Block num will be freed after request is swapped out
};

// Define event start and end point
enum TraceEventPhase { Begin, End };

struct TraceEventRecord {
  std::string event_name;
  TraceEventType event_type;
  std::string process_name;
  std::string thread_name;
  TraceEventPhase phase;
  std::time_t time_ns;
};

class TraceEventRecorder {
 public:
  TraceEventRecorder();
  ~TraceEventRecorder();

  void Record(TraceEventRecord& record) {
    {
      std::unique_lock<std::mutex> lock{mutex_};
      record_queue_.push(record);
    }
    cv_.notify_one();
  }

  static TraceEventRecorder* GetInstance() {
    static TraceEventRecorder recorder;
    return &recorder;
  }

 private:
  void Process();

 private:
  std::atomic<bool> destroying_;

  std::condition_variable cv_;
  std::mutex mutex_;
  std::queue<TraceEventRecord> record_queue_;

  // The process thread.
  std::thread recorder_thread_;
};

#ifdef ENABLE_RECORD_EVENT

#  define RECORD_TRACE_EVENT(e_name, e_type, p_name, t_name, ph, t) \
    {                                                               \
      TraceEventRecord record;                                      \
      record.event_name = e_name;                                   \
      record.event_type = e_type;                                   \
      record.process_name = p_name;                                 \
      record.thread_name = t_name;                                  \
      record.phase = ph;                                            \
      record.time_ns = t;                                           \
      TraceEventRecorder::GetInstance()->Record(record);            \
    }

#  define RECORD_TRACE_EVENT_BEGIN(name, type, p_name, t_name)                         \
    {                                                                                  \
      std::time_t time_ns = ProfileTimer::GetCurrentTimeInNs();                        \
      RECORD_TRACE_EVENT(name, type, p_name, t_name, TraceEventPhase::Begin, time_ns); \
    }

#  define RECORD_TRACE_EVENT_END(name, type, p_name, t_name)                         \
    {                                                                                \
      std::time_t time_ns = ProfileTimer::GetCurrentTimeInNs();                      \
      RECORD_TRACE_EVENT(name, type, p_name, t_name, TraceEventPhase::End, time_ns); \
    }

#  define RECORD_TRACE_EVENT_TAG(name, type, p_name, t_name)                                 \
    {                                                                                        \
      std::time_t time_ns = ProfileTimer::GetCurrentTimeInNs();                              \
      RECORD_TRACE_EVENT(name, type, p_name, t_name, TraceEventPhase::Begin, time_ns);       \
      RECORD_TRACE_EVENT(name, type, p_name, t_name, TraceEventPhase::End, time_ns + 20000); \
    }

#else
#  define RECORD_TRACE_EVENT(e_name, e_type, p_name, t_name, ph, t)
#  define RECORD_TRACE_EVENT_BEGIN(name, type, p_name, t_name)
#  define RECORD_TRACE_EVENT_END(name, type, p_name, t_name)
#  define RECORD_TRACE_EVENT_TAG(name, type, p_name, t_name)
#endif

#define TRACE_PROCESS_NAME_METRICS ("-1")

#define TRACE_THREAD_NAME_PREFILL_DECODE ("Prefill/Decode")
#define TRACE_THREAD_NAME_SWAP ("SwapOut/SwapIn")
#define TRACE_THREAD_NAME_INPUT_TOKEN_NUM ("InputTokenNum")
#define TRACE_THREAD_NAME_FORWARD_TOKEN_NUM ("ForwardTokenNum")
#define TRACE_THREAD_NAME_FORWARD_TOKEN_PER_SEC ("ForwardTokenNumPerSec")
#define TRACE_THREAD_NAME_USABLE_BLK_NUM ("UsableBlockNum")
#define TRACE_THREAD_NAME_FUTURE_BLK_NUM ("FutureBlockNum")

}  // namespace ksana_llm
