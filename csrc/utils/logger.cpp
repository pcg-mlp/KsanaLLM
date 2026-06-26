/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "logger.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace ksana_llm {

static std::vector<std::string>& GetDetailLevelsInternal() {
  static std::vector<std::string> detail_levels;
  return detail_levels;
}

static std::string& GetLogFilenameStorage() {
  static std::string log_filename_storage;
  return log_filename_storage;
}

static LogConfig s_log_config;
static std::atomic<bool> s_log_config_set{false};
static std::mutex s_log_rotation_mutex;

// 日志文件描述符管理
static int s_log_fd = -1;
static std::atomic<ino_t> s_log_inode{0};
static std::mutex s_log_fd_mutex;

// 早期日志缓冲区（在配置设置前暂存日志）
static std::vector<std::string>& GetEarlyLogBuffer() {
  static std::vector<std::string> buffer;
  return buffer;
}
static std::mutex s_early_log_mutex;

// Get user-specified log basename from KLLM_LOG_FILE environment variable.
// Strips directory prefix and .log suffix. Returns "ksana_llm" if not set.
static std::string GetUserLogBasename() {
  const char* kllm_log_file = std::getenv("KLLM_LOG_FILE");
  if (kllm_log_file == nullptr || kllm_log_file[0] == '\0') {
    return "ksana_llm";
  }
  std::string path(kllm_log_file);
  // Extract filename part (remove directory).
  size_t last_slash = path.find_last_of('/');
  std::string basename = (last_slash != std::string::npos) ? path.substr(last_slash + 1) : path;
  // Strip .log suffix.
  const std::string log_suffix = ".log";
  if (basename.size() > log_suffix.size() &&
      basename.compare(basename.size() - log_suffix.size(), log_suffix.size(), log_suffix) == 0) {
    basename = basename.substr(0, basename.size() - log_suffix.size());
  }
  return basename;
}

static std::string GetLogBasename() {
  const char* world_size_env = std::getenv("WORLD_SIZE");
  const char* node_rank = std::getenv("NODE_RANK");
  if (world_size_env != nullptr && world_size_env[0] != '\0' && std::stoi(world_size_env) > 1 && node_rank != nullptr &&
      node_rank[0] != '\0') {
    return GetUserLogBasename() + "_rank" + node_rank;
  }
  return GetUserLogBasename();
}

// 打开或重新打开日志文件
static void OpenLogFile() {
  std::lock_guard<std::mutex> lock(s_log_fd_mutex);

  if (s_log_fd >= 0) {
    close(s_log_fd);
  }

  std::string log_file = GetLogFile();

  // 确保日志目录存在
  std::filesystem::path log_path(log_file);
  if (log_path.has_parent_path()) {
    std::filesystem::create_directories(log_path.parent_path());
  }

  s_log_fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);

  if (s_log_fd >= 0) {
    struct stat st;
    if (fstat(s_log_fd, &st) == 0) {
      s_log_inode.store(st.st_ino, std::memory_order_relaxed);
    } else {
      fprintf(stderr, "[KSANA] WARNING: fstat failed for %s: %s\n", log_file.c_str(), strerror(errno));
    }
  } else {
    fprintf(stderr, "[KSANA] ERROR: Failed to open log file %s: %s\n", log_file.c_str(), strerror(errno));
  }
}

// 检查文件是否被切割（inode 变化），如果是则重新打开
static void CheckAndReopenLogFile() {
  std::string log_file = GetLogFile();
  struct stat st;

  if (stat(log_file.c_str(), &st) == 0) {
    // s_log_inode 是原子变量，读取是线程安全的
    // 最坏情况：多个线程同时检测到变化并调用 OpenLogFile，但 OpenLogFile 内部有锁保护
    if (st.st_ino != s_log_inode.load(std::memory_order_relaxed)) {
      OpenLogFile();
    }
  }
}

// 写入日志到文件
static void WriteToLogFile(const char* message, size_t len) {
  std::lock_guard<std::mutex> lock(s_log_fd_mutex);

  if (s_log_fd < 0) {
    return;
  }

  ssize_t written = write(s_log_fd, message, len);
  if (written < 0) {
    // 写入失败，可能文件被删除，尝试重新打开
    std::string log_file = GetLogFile();
    close(s_log_fd);
    s_log_fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (s_log_fd >= 0) {
      write(s_log_fd, message, len);
    } else {
      fprintf(stderr, "[KSANA] ERROR: Failed to reopen log file %s: %s\n", log_file.c_str(), strerror(errno));
    }
  }
}

// 共享的日志文件初始化标志
static bool s_log_file_initialized = false;
static std::mutex s_log_init_mutex;

static void EnsureLogFileInitialized() {
  std::lock_guard<std::mutex> lock(s_log_init_mutex);
  if (s_log_file_initialized) {
    return;
  }

  OpenLogFile();

  // 写入早期缓存的日志
  {
    std::lock_guard<std::mutex> early_lock(s_early_log_mutex);
    auto& buffer = GetEarlyLogBuffer();
    for (const auto& early_log : buffer) {
      // 直接写入，避免递归锁
      if (s_log_fd >= 0) {
        write(s_log_fd, early_log.c_str(), early_log.size());
      }
    }
    buffer.clear();
    buffer.shrink_to_fit();
  }

  s_log_file_initialized = true;
}

// 外部接口
std::vector<std::string>& GetDetailLevels() { return GetDetailLevelsInternal(); }
void ClearDetailLevels() { GetDetailLevelsInternal().clear(); }
void AddDetailLevel(const std::string& level) { GetDetailLevelsInternal().push_back(level); }

void SetLogConfig(const LogConfig& config) {
  s_log_config = config;
  s_log_config_set.store(true, std::memory_order_release);

  // 更新日志文件路径到配置的目录
  std::string& storage = GetLogFilenameStorage();

  if (!storage.empty()) {
    // 从旧文件名提取 filename，用新的 log_dir 重新生成
    size_t last_slash = storage.find_last_of('/');
    std::string filename = (last_slash != std::string::npos) ? storage.substr(last_slash + 1) : storage;

    // 确保新目录存在
    std::filesystem::create_directories(config.log_dir);

    // 更新存储的路径（确保以 / 结尾）
    std::string log_dir = config.log_dir;
    if (!log_dir.empty() && log_dir.back() != '/') {
      log_dir += '/';
    }
    storage = log_dir + filename;
  }

  // 如果日志文件已经打开，需要重新打开到配置的路径
  if (s_log_fd >= 0) {
    OpenLogFile();
    KLLM_LOG_INFO << "Log file switched to: " << storage;
  }
}

const LogConfig& GetLogConfig() { return s_log_config; }

void SetLogFile(const std::string& postfix) {
  // 使用配置的日志目录，默认 "log/"
  std::string log_dir = s_log_config_set.load(std::memory_order_acquire) ? s_log_config.log_dir : "log/";

  // Get user-specified basename from KLLM_LOG_FILE, default "ksana_llm".
  std::string basename = GetUserLogBasename();

  // In PP mode, insert rank suffix between basename and role postfix.
  std::string rank_suffix;
  const char* world_size_env = std::getenv("WORLD_SIZE");
  const char* node_rank = std::getenv("NODE_RANK");
  if (world_size_env != nullptr && world_size_env[0] != '\0' && std::stoi(world_size_env) > 1 && node_rank != nullptr &&
      node_rank[0] != '\0') {
    rank_suffix = std::string("_rank") + node_rank;
  }

  // Build log filename: basename + rank_suffix + postfix + ".log".
  std::string log_name;
  if (postfix.empty() || postfix == "_python") {
    log_name = basename + rank_suffix + "_python.log";
  } else if (postfix.size() > 1 && postfix[0] == '_') {
    log_name = basename + rank_suffix + postfix + ".log";
  } else {
    log_name = basename + rank_suffix + "_" + postfix + ".log";
  }

  std::string log_filename = log_dir + log_name;
  GetLogFilenameStorage() = log_filename;
}

std::string GetLogFile() {
  const std::string& storage = GetLogFilenameStorage();
  if (!storage.empty()) {
    return storage;
  }
  std::string log_dir = s_log_config_set.load(std::memory_order_acquire) ? s_log_config.log_dir : "log/";
  return log_dir + GetLogBasename() + ".log";
}

size_t GetLogMaxFileSize() {
  if (s_log_config_set.load(std::memory_order_acquire)) {
    return s_log_config.max_file_size;
  }
  return 536870912;  // 默认 512MB
}

int GetLogMaxBackupCount() {
  if (s_log_config_set.load(std::memory_order_acquire)) {
    return s_log_config.max_backup_count;
  }
  return 2;  // 默认 2 个备份
}

void RotateLogFiles() {
  std::lock_guard<std::mutex> lock(s_log_rotation_mutex);

  std::string log_file = GetLogFile();
  size_t max_size = GetLogMaxFileSize();
  int max_backups = GetLogMaxBackupCount();

  struct stat st;
  if (stat(log_file.c_str(), &st) != 0) {
    return;
  }

  if (static_cast<size_t>(st.st_size) < max_size) {
    return;
  }

  std::error_code ec;

  // 1. 删除最旧的备份（如果存在）
  std::string oldest = log_file + "." + std::to_string(max_backups);
  std::filesystem::remove(oldest, ec);  // 忽略错误

  // 2. 依次重命名 .N-1 -> .N（只重命名存在的文件）
  for (int i = max_backups - 1; i >= 1; --i) {
    std::string src = log_file + "." + std::to_string(i);
    std::string dst = log_file + "." + std::to_string(i + 1);
    if (std::filesystem::exists(src)) {
      std::filesystem::rename(src, dst, ec);
    }
  }

  // 3. 当前文件 -> .1
  std::filesystem::rename(log_file, log_file + ".1", ec);

  // 重新打开日志文件（创建新文件）
  OpenLogFile();
}

// loguru 日志回调，写入日志并检查是否需要滚动
void log_file_callback(void* user_data, const loguru::Message& message) {
  std::string log_line = std::string(message.preamble) + message.message + "\n";

  // 延迟初始化：只有在配置设置后才打开日志文件
  if (!s_log_config_set.load(std::memory_order_acquire)) {
    // 配置还未设置，暂存到内存缓冲区
    std::lock_guard<std::mutex> lock(s_early_log_mutex);
    auto& buffer = GetEarlyLogBuffer();
    // 限制缓冲区大小，避免内存问题（最多 1000 条）
    if (buffer.size() < 1000) {
      buffer.push_back(log_line);
    }
    return;
  }

  EnsureLogFileInitialized();

  WriteToLogFile(log_line.c_str(), log_line.size());

  // 每隔一定数量的日志检查一次是否需要切割
  static std::atomic<int> log_count{0};
  constexpr int CHECK_INTERVAL = 100;  // 每 100 条日志检查一次

  if (log_count.fetch_add(1, std::memory_order_relaxed) >= CHECK_INTERVAL) {
    log_count.store(0, std::memory_order_relaxed);
    // 先检查是否被其他进程切割
    CheckAndReopenLogFile();
    // 再检查是否需要主动切割
    RotateLogFiles();
  }
}

void details_log_handler(void* user_data, const loguru::Message& message) {
  const auto& detail_levels = GetDetailLevels();

  // 检查是否匹配 detail 日志级别
  bool matches = false;
  for (const auto& log_level : detail_levels) {
    if (message.message && std::string_view(message.message).find(log_level) != std::string_view::npos) {
      matches = true;
      break;
    }
  }

  if (!matches) {
    return;
  }

  // 格式化日志消息
  std::string log_line = std::string(message.preamble) + message.message + "\n";

  // 延迟初始化：只有在配置设置后才写入日志文件
  if (!s_log_config_set.load(std::memory_order_acquire)) {
    // 配置还未设置，暂存到内存缓冲区
    std::lock_guard<std::mutex> lock(s_early_log_mutex);
    auto& buffer = GetEarlyLogBuffer();
    if (buffer.size() < 1000) {
      buffer.push_back(log_line);
    }
    return;
  }

  // 确保日志文件已初始化
  EnsureLogFileInitialized();

  // 使用统一的日志写入，确保使用正确的路径
  WriteToLogFile(log_line.c_str(), log_line.size());

  // 每隔一定数量的日志检查一次是否需要切割
  static std::atomic<int> log_count{0};
  constexpr int CHECK_INTERVAL = 100;

  if (log_count.fetch_add(1, std::memory_order_relaxed) >= CHECK_INTERVAL) {
    log_count.store(0, std::memory_order_relaxed);
    CheckAndReopenLogFile();
    RotateLogFiles();
  }
}

IntervalLogger::IntervalLogger(uint64_t interval_ms) : interval_ms_(interval_ms), last_time_ms_(GetCurrentTimeInMs()) {}

bool IntervalLogger::ShouldLog() {
  const uint64_t current_time_ms = GetCurrentTimeInMs();
  if (current_time_ms - last_time_ms_ < interval_ms_) {
    return false;
  }
  last_time_ms_ = current_time_ms;
  return true;
}

}  // namespace ksana_llm
