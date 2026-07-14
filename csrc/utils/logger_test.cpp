/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * Logger unit tests - covers all functions in logger.cpp
 ==============================================================================*/

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#include "configure/log_config.h"
#include "loguru.hpp"
#include "test.h"
#include "utils/logger.h"

namespace ksana_llm {

class LoggerTest : public ::testing::Test {
 protected:
  std::string test_log_dir_;
  std::string test_log_file_;

  void SetUp() override {
    // 创建临时测试目录
    test_log_dir_ = "/tmp/ksana_logger_test_" + std::to_string(getpid()) + "/";
    std::filesystem::create_directories(test_log_dir_);
    test_log_file_ = test_log_dir_ + "test.log";

    // 重置 logger 状态
    ClearDetailLevels();
  }

  void TearDown() override {
    // 清理测试目录
    std::error_code ec;
    std::filesystem::remove_all(test_log_dir_, ec);

    // 清理环境变量
    unsetenv("KLLM_LOG_LEVEL");
  }

  // 辅助函数：读取文件内容
  std::string ReadFileContents(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      return "";
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  }

  // 辅助函数：创建指定大小的文件
  void CreateFileWithSize(const std::string& path, size_t size) {
    std::ofstream file(path, std::ios::binary);
    std::string content(size, 'X');
    file.write(content.c_str(), size);
  }
};

// ============== SetLogFile 和 GetLogFile 测试 ==============

TEST_F(LoggerTest, SetLogFile_Engine) {
  SetLogFile("_engine");
  std::string log_file = GetLogFile();
  EXPECT_TRUE(log_file.find("ksana_engine.log") != std::string::npos);
}

TEST_F(LoggerTest, SetLogFile_Executor) {
  SetLogFile("_executor0");
  EXPECT_TRUE(GetLogFile().find("ksana_executor0.log") != std::string::npos);

  SetLogFile("_executor1");
  EXPECT_TRUE(GetLogFile().find("ksana_executor1.log") != std::string::npos);
}

TEST_F(LoggerTest, SetLogFile_Python) {
  SetLogFile("_python");
  EXPECT_TRUE(GetLogFile().find("ksana_python.log") != std::string::npos);

  SetLogFile("");
  EXPECT_TRUE(GetLogFile().find("ksana_python.log") != std::string::npos);
}

// ============== ParseFileSize 测试 ==============

TEST_F(LoggerTest, ParseFileSize_Bytes) {
  // 纯数字（字节）
  EXPECT_EQ(ParseFileSize("1024"), 1024);
  EXPECT_EQ(ParseFileSize("536870912"), 536870912);
}

TEST_F(LoggerTest, ParseFileSize_KB) {
  // KB 格式
  EXPECT_EQ(ParseFileSize("1kb"), 1024);
  EXPECT_EQ(ParseFileSize("1KB"), 1024);
  EXPECT_EQ(ParseFileSize("100k"), 100 * 1024);
  EXPECT_EQ(ParseFileSize("100K"), 100 * 1024);
}

TEST_F(LoggerTest, ParseFileSize_MB) {
  // MB 格式
  EXPECT_EQ(ParseFileSize("1mb"), 1024 * 1024);
  EXPECT_EQ(ParseFileSize("1MB"), 1024 * 1024);
  EXPECT_EQ(ParseFileSize("512m"), 512 * 1024 * 1024);
  EXPECT_EQ(ParseFileSize("512M"), 512 * 1024 * 1024);
  EXPECT_EQ(ParseFileSize("100mb"), 100 * 1024 * 1024);
}

TEST_F(LoggerTest, ParseFileSize_GB) {
  // GB 格式
  EXPECT_EQ(ParseFileSize("1gb"), static_cast<size_t>(1024) * 1024 * 1024);
  EXPECT_EQ(ParseFileSize("1GB"), static_cast<size_t>(1024) * 1024 * 1024);
  EXPECT_EQ(ParseFileSize("2g"), static_cast<size_t>(2) * 1024 * 1024 * 1024);
}

TEST_F(LoggerTest, ParseFileSize_WithSpaces) {
  // 带空格
  EXPECT_EQ(ParseFileSize("100 mb"), 100 * 1024 * 1024);
  EXPECT_EQ(ParseFileSize(" 512 MB "), 512 * 1024 * 1024);
}

TEST_F(LoggerTest, ParseFileSize_Decimal) {
  // 小数
  EXPECT_EQ(ParseFileSize("1.5mb"), static_cast<size_t>(1.5 * 1024 * 1024));
  EXPECT_EQ(ParseFileSize("0.5gb"), static_cast<size_t>(0.5 * 1024 * 1024 * 1024));
}

TEST_F(LoggerTest, ParseFileSize_Default) {
  // 空字符串返回默认值 512MB
  EXPECT_EQ(ParseFileSize(""), 536870912);
}

TEST_F(LoggerTest, ParseFileSize_ConfigIntegration) {
  // 测试配置集成 - 使用人类可读格式设置 LogConfig
  LogConfig config;
  config.log_dir = test_log_dir_;
  config.max_file_size = ParseFileSize("256mb");
  config.max_backup_count = 3;

  SetLogFile("_engine");
  SetLogConfig(config);

  const LogConfig& retrieved = GetLogConfig();
  EXPECT_EQ(retrieved.max_file_size, 256 * 1024 * 1024);
}

// ============== SetLogConfig 和 GetLogConfig 测试 ==============

TEST_F(LoggerTest, SetLogConfig_Basic) {
  LogConfig config;
  config.log_dir = test_log_dir_;
  config.max_file_size = 1024 * 1024;  // 1MB
  config.max_backup_count = 3;

  SetLogFile("_engine");
  SetLogConfig(config);

  const LogConfig& retrieved = GetLogConfig();
  EXPECT_EQ(retrieved.log_dir, test_log_dir_);
  EXPECT_EQ(retrieved.max_file_size, 1024 * 1024);
  EXPECT_EQ(retrieved.max_backup_count, 3);

  // 验证日志路径已更新
  std::string log_file = GetLogFile();
  EXPECT_TRUE(log_file.find(test_log_dir_) != std::string::npos);
}

TEST_F(LoggerTest, GetLogMaxFileSize_Default) {
  size_t size = GetLogMaxFileSize();
  EXPECT_GT(size, 0);
}

TEST_F(LoggerTest, GetLogMaxBackupCount_Default) {
  int count = GetLogMaxBackupCount();
  EXPECT_GT(count, 0);
}

// ============== DetailLevels 管理测试 ==============

TEST_F(LoggerTest, DetailLevels_AddAndGet) {
  ClearDetailLevels();
  EXPECT_TRUE(GetDetailLevels().empty());

  AddDetailLevel("ATTENTION");
  AddDetailLevel("SCHEDULER");

  auto& levels = GetDetailLevels();
  EXPECT_EQ(levels.size(), 2);
  EXPECT_EQ(levels[0], "ATTENTION");
  EXPECT_EQ(levels[1], "SCHEDULER");
}

TEST_F(LoggerTest, DetailLevels_Clear) {
  AddDetailLevel("MOE");
  AddDetailLevel("MODEL");
  EXPECT_FALSE(GetDetailLevels().empty());

  ClearDetailLevels();
  EXPECT_TRUE(GetDetailLevels().empty());
}

// ============== RotateLogFiles 测试 ==============

TEST_F(LoggerTest, RotateLogFiles_NoRotationNeeded) {
  LogConfig config;
  config.log_dir = test_log_dir_;
  config.max_file_size = 1024 * 1024;  // 1MB
  config.max_backup_count = 3;

  SetLogFile("_test");
  SetLogConfig(config);

  std::string log_file = GetLogFile();

  // 创建一个小文件（小于 max_file_size）
  CreateFileWithSize(log_file, 1024);  // 1KB

  // 调用 RotateLogFiles，不应发生滚动
  RotateLogFiles();

  // 原文件应该还在
  EXPECT_TRUE(std::filesystem::exists(log_file));
  // 不应该有 .1 备份
  EXPECT_FALSE(std::filesystem::exists(log_file + ".1"));
}

TEST_F(LoggerTest, RotateLogFiles_SingleRotation) {
  LogConfig config;
  config.log_dir = test_log_dir_;
  config.max_file_size = 1024;  // 1KB
  config.max_backup_count = 3;

  SetLogFile("_rotate");
  SetLogConfig(config);

  std::string log_file = GetLogFile();

  // 创建一个大于 max_file_size 的文件
  CreateFileWithSize(log_file, 2048);  // 2KB

  // 调用 RotateLogFiles
  RotateLogFiles();

  // 原文件应该被重命名为 .1
  EXPECT_TRUE(std::filesystem::exists(log_file + ".1"));
}

TEST_F(LoggerTest, RotateLogFiles_MultipleRotations) {
  LogConfig config;
  config.log_dir = test_log_dir_;
  config.max_file_size = 1024;  // 1KB
  config.max_backup_count = 3;

  SetLogFile("_multi");
  SetLogConfig(config);

  std::string log_file = GetLogFile();

  // 模拟多次滚动
  for (int i = 0; i < 5; ++i) {
    CreateFileWithSize(log_file, 2048);
    RotateLogFiles();
  }

  // 应该有 .1, .2, .3 备份，但不应该有 .4（超过 max_backup_count）
  EXPECT_TRUE(std::filesystem::exists(log_file + ".1"));
  EXPECT_TRUE(std::filesystem::exists(log_file + ".2"));
  EXPECT_TRUE(std::filesystem::exists(log_file + ".3"));
  EXPECT_FALSE(std::filesystem::exists(log_file + ".4"));
}

// ============== IntervalLogger 测试 ==============

TEST_F(LoggerTest, IntervalLogger_Basic) {
  IntervalLogger logger(10);  // 10ms 间隔

  // 构造函数记录当前时间，所以第一次调用如果间隔未到会返回 false
  // 等待超过间隔时间后调用应该返回 true
  std::this_thread::sleep_for(std::chrono::milliseconds(15));

  bool first = logger.ShouldLog();
  // 立即再次调用应该返回 false（间隔未到）
  bool second = logger.ShouldLog();

  EXPECT_TRUE(first);
  EXPECT_FALSE(second);
}

TEST_F(LoggerTest, IntervalLogger_AfterWait) {
  IntervalLogger logger(50);  // 50ms 间隔

  logger.ShouldLog();  // 第一次调用

  // 等待超过间隔时间
  std::this_thread::sleep_for(std::chrono::milliseconds(60));

  // 现在应该返回 true
  EXPECT_TRUE(logger.ShouldLog());
}

// ============== 日志写入集成测试 ==============

TEST_F(LoggerTest, LogWriting_BasicInfo) {
  LogConfig config;
  config.log_dir = test_log_dir_;
  config.max_file_size = 10 * 1024 * 1024;
  config.max_backup_count = 5;

  setenv("KLLM_LOG_LEVEL", "INFO", 1);
  SetLogFile("_info");
  SetLogConfig(config);

  // 强制初始化 loguru
  InitLoguru(true, "_info");

  KLLM_LOG_INFO << "Test INFO message";

  // 等待日志写入
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::string log_file = GetLogFile();
  std::string contents = ReadFileContents(log_file);

  EXPECT_TRUE(contents.find("Test INFO message") != std::string::npos);
}

TEST_F(LoggerTest, LogWriting_DetailLevels) {
  LogConfig config;
  config.log_dir = test_log_dir_;
  config.max_file_size = 10 * 1024 * 1024;
  config.max_backup_count = 5;

  setenv("KLLM_LOG_LEVEL", "ATTENTION,SCHEDULER", 1);
  SetLogFile("_detail");
  SetLogConfig(config);

  InitLoguru(true, "_detail");

  KLLM_LOG_ATTENTION << "Test ATTENTION message";
  KLLM_LOG_SCHEDULER << "Test SCHEDULER message";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::string log_file = GetLogFile();
  std::string contents = ReadFileContents(log_file);

  EXPECT_TRUE(contents.find("ATTENTION") != std::string::npos);
  EXPECT_TRUE(contents.find("SCHEDULER") != std::string::npos);
}

TEST_F(LoggerTest, LogWriting_DebugLevel) {
  LogConfig config;
  config.log_dir = test_log_dir_;
  config.max_file_size = 10 * 1024 * 1024;
  config.max_backup_count = 5;

  setenv("KLLM_LOG_LEVEL", "DEBUG", 1);
  SetLogFile("_debug");
  SetLogConfig(config);

  InitLoguru(true, "_debug");

  KLLM_LOG_DEBUG << "Test DEBUG message";
  KLLM_LOG_INFO << "Test INFO message";
  KLLM_LOG_MOE << "Test MOE message";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::string log_file = GetLogFile();
  std::string contents = ReadFileContents(log_file);

  // DEBUG 级别应该包含所有日志
  EXPECT_TRUE(contents.find("DEBUG") != std::string::npos);
  EXPECT_TRUE(contents.find("INFO") != std::string::npos);
  EXPECT_TRUE(contents.find("MOE") != std::string::npos);
}

// ============== GetLogLevels 测试 ==============

TEST_F(LoggerTest, GetLogLevels_Single) {
  setenv("KLLM_LOG_LEVEL", "INFO", 1);
  auto levels = GetLogLevels();
  EXPECT_EQ(levels.size(), 1);
  EXPECT_EQ(levels[0], "INFO");
}

TEST_F(LoggerTest, GetLogLevels_Multiple) {
  setenv("KLLM_LOG_LEVEL", "INFO,ATTENTION,COMMUNICATION", 1);
  auto levels = GetLogLevels();
  EXPECT_EQ(levels.size(), 3);
  EXPECT_EQ(levels[0], "INFO");
  EXPECT_EQ(levels[1], "ATTENTION");
  EXPECT_EQ(levels[2], "COMMUNICATION");
}

}  // namespace ksana_llm
