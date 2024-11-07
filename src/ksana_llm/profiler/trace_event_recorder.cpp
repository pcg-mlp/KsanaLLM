/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <memory>
#include <unordered_map>

#include "ksana_llm/profiler/trace_event_recorder.h"
#include "ksana_llm/utils/logger.h"

using namespace ksana_llm;

static std::string GetTraceEventRecordFile() {
  std::string default_event_record_file =
      "log/ksana_event_record" + ProfileTimer::GetCurrentTimeInStr("%Y%m%d_%H%M%S") + std::string(".json");
  const char* env_event_record_file = std::getenv("KLLM_EVENT_RECORD_FILE");
  return env_event_record_file ? env_event_record_file : default_event_record_file;
}

TraceEventRecorder::TraceEventRecorder() {
  recorder_thread_ = std::thread(&TraceEventRecorder::Process, this);
  destroying_.store(false);
}

TraceEventRecorder::~TraceEventRecorder() {
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    destroying_.store(true);
  }
  cv_.notify_all();
  if (recorder_thread_.joinable()) {
    recorder_thread_.join();
  }
}

void TraceEventRecorder::Process() {
  std::unordered_map<TraceEventType, std::string> color_map;
  // Valid colors are defined in
  // https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
  color_map[TraceEventType::SchedBegin] = "black";
  color_map[TraceEventType::DropReq] = "terrible";
  color_map[TraceEventType::FinishReq] = "black";

  color_map[TraceEventType::Prefill] = "thread_state_running";
  color_map[TraceEventType::Decode] = "olive";
  color_map[TraceEventType::SwapOut] = "bad";
  color_map[TraceEventType::SwapIn] = "grey";
  color_map[TraceEventType::InputTokenNum] = "rail_response";
  color_map[TraceEventType::ForwardTokenNum] = "startup";
  color_map[TraceEventType::TokenNumPerSec] = "yellow";
  color_map[TraceEventType::UsableBlockNum] = "rail_load";
  color_map[TraceEventType::FutureBlockNum] = "bad";

  /*
  Example of a file:
  {
    "displayTimeUnit": "ns",
    "traceEvents": [
    {"name": "P1", "ph": "B", "pid": "0", "tid": "0", "ts": 0, "cname": "yellow" },
    {"name": "P1", "ph": "E", "pid": "0", "tid": "0", "ts": 1, "cname": "yellow" },
    ...
    {"name": "P1", "ph": "E", "pid": "2", "tid": "0", "ts": 1, "cname": "yellow" }
    ]
  }
  */
  bool is_first_record = true;
  // The output file stream.
  std::ofstream recorder_stream;
  std::string trace_filename = GetTraceEventRecordFile();
  recorder_stream.open(trace_filename.c_str(), std::ios::app);
  if (!recorder_stream.is_open()) {
    KLLM_LOG_WARNING << "Failed to open trace file: " << trace_filename;
    return;
  }
  recorder_stream << "{ \n \"displayTimeUnit\": \"ns\",\n \"traceEvents\": [" << std::endl;
  while (!destroying_) {
    std::unique_lock<std::mutex> lock{mutex_};
    cv_.wait(lock, [this] { return destroying_.load() || !record_queue_.empty(); });
    while (!record_queue_.empty()) {
      auto record = std::move(record_queue_.front());
      if (is_first_record) {
        is_first_record = false;
      } else {
        recorder_stream << ",\n";
      }
      // Note: because chrome://tracing display problem, value of ts should be changed from ns to us
      recorder_stream << "  {\"name\": \"" << record.event_name << "\", \"ph\": \""
                      << (record.phase == TraceEventPhase::Begin ? "B" : "E") << "\", \"pid\": \""
                      << record.process_name << "\", \"tid\": \"" << record.thread_name
                      << "\", \"ts\": " << record.time_ns / 1000 << ", \"cname\": \"" << color_map[record.event_type]
                      << "\" }";
      record_queue_.pop();
    }
    recorder_stream.flush();
  }
  recorder_stream << "\n]\n}" << std::endl;
  recorder_stream.close();
}
