/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/profiler/writer.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

static std::string GetStatFile() {
  const char* default_stat_file = "log/ksana_stat.log";
  const char* env_stat_file = std::getenv("KLLM_STAT_FILE");
  return env_stat_file ? env_stat_file : default_stat_file;
}

ProfileWriter::ProfileWriter() {
  profiler_file_ = GetStatFile();
  profiler_stream_.open(profiler_file_.c_str(), std::ios::app);
}

ProfileWriter::~ProfileWriter() {
  if (profiler_stream_.is_open()) {
    profiler_stream_.close();
  }
}

// Write message to disk file.
Status ProfileWriter::Write(const std::string& message) {
  profiler_stream_ << message << std::endl;
  return Status();
}

Status ProfileWriter::Write(const std::vector<std::string>& messages) {
  for (auto& message : messages) {
    Write(message);
  }
  profiler_stream_ << std::flush;
  return Status();
}

}  // namespace ksana_llm
