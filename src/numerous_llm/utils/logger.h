/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <unordered_map>

#define LOGURU_USE_FMTLIB 1
#define LOGURU_WITH_STREAMS 1
#include "loguru.hpp"

namespace numerous_llm {

// Log level.
enum Level { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, FATAL = 4 };

// Log name to level.
static std::unordered_map<Level, std::string> log_level_to_name = {
    {DEBUG, "DEBUG"}, {INFO, "INFO"}, {WARNING, "WARNING"}, {ERROR, "ERROR"}, {FATAL, "FATAL"}};

// Get log level from environment, this function called only once.
static Level GetLogLevel() {
  const char* default_log_level = "INFO";
  const char* env_log_level = std::getenv("NLLM_LOG_LEVEL");
  std::string log_level_str = env_log_level ? env_log_level : default_log_level;

  std::unordered_map<std::string, Level> log_name_to_level = {
      {"DEBUG", DEBUG}, {"INFO", INFO}, {"WARNING", WARNING}, {"ERROR", ERROR}, {"FATAL", FATAL}};

  Level level = Level::INFO;
  if (log_name_to_level.find(log_level_str) != log_name_to_level.end()) {
    level = log_name_to_level[log_level_str];
  }

  return level;
}

// Get log filename from environment, called once.
static std::string GetLogFile() {
  const char* default_log_file = "log/numerous_llm.log";
  const char* env_log_file = std::getenv("NLLM_LOG_FILE");
  return env_log_file ? env_log_file : default_log_file;
}

// Get name from log level.
static std::string GetLevelName(const Level level) { return log_level_to_name[level]; }

// Init logrun instance.
inline void InitLoguru() {
  Level log_level = GetLogLevel();

  loguru::Verbosity verbosity = loguru::Verbosity_MAX;
  if (log_level >= Level::DEBUG) {
    verbosity = loguru::Verbosity_MAX;
  } else if (log_level == Level::INFO) {
    verbosity = loguru::Verbosity_INFO;
  } else if (log_level == Level::WARNING) {
    verbosity = loguru::Verbosity_WARNING;
  } else if (log_level == Level::ERROR) {
    verbosity = loguru::Verbosity_ERROR;
  } else if (log_level == Level::FATAL) {
    verbosity = loguru::Verbosity_FATAL;
  }
  loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
  loguru::add_file(GetLogFile().c_str(), loguru::Append, verbosity);
}

#define NO_CC_IF if  // For CodeCC compatibility.

#define NLLM_LOG_DEBUG LOG_S(1)
#define NLLM_LOG_INFO LOG_S(INFO)
#define NLLM_LOG_WARNING LOG_S(WARNING)
#define NLLM_LOG_ERROR LOG_S(ERROR)
#define NLLM_LOG_FATAL LOG_S(FATAL)

[[noreturn]] inline void ThrowRuntimeError(const char* const file, int const line, std::string const& info = "") {
  throw std::runtime_error(std::string("[NLLM][ERROR] ") + info + " Assertion fail: " + file + ":" +
                           std::to_string(line) + " \n");
}

inline void CheckAssert(bool result, const char* const file, int const line, std::string const& info = "") {
  if (!result) {
    ThrowRuntimeError(file, line, info);
  }
}

#define NLLM_CHECK(val) CheckAssert(val, __FILE__, __LINE__)
#define NLLM_CHECK_WITH_INFO(val, info)                                    \
  do {                                                                     \
    bool is_valid_val = (val);                                             \
    if (!is_valid_val) {                                                   \
      numerous_llm::CheckAssert(is_valid_val, __FILE__, __LINE__, (info)); \
    }                                                                      \
  } while (0)

#define NLLM_THROW(info) ThrowRuntimeError(__FILE__, __LINE__, info)

}  // namespace numerous_llm
