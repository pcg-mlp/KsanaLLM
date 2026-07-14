/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cctype>
#include <cstddef>
#include <string>

namespace ksana_llm {

// 日志配置结构
struct LogConfig {
  // 日志目录路径，默认 "log/"
  std::string log_dir = "log/";

  // 单个日志文件最大大小（字节），默认 512MB
  size_t max_file_size = 536870912;

  // 最大备份文件数，默认 2
  int max_backup_count = 2;
};

// 解析人类可读的文件大小格式，如 "100mb", "1gb", "512KB"
// 支持: B, KB, MB, GB (大小写不敏感)
// 如果是纯数字，则直接作为字节数
inline size_t ParseFileSize(const std::string& size_str) {
  if (size_str.empty()) {
    return 536870912;  // 默认 512MB
  }

  // 找到数字部分的结束位置
  size_t num_end = 0;
  while (num_end < size_str.size() && (std::isdigit(size_str[num_end]) || size_str[num_end] == '.')) {
    ++num_end;
  }

  if (num_end == 0) {
    return 536870912;  // 无法解析，返回默认值
  }

  double value = std::stod(size_str.substr(0, num_end));

  // 获取单位部分并转换为大写
  std::string unit;
  for (size_t i = num_end; i < size_str.size(); ++i) {
    if (!std::isspace(size_str[i])) {
      unit += std::toupper(size_str[i]);
    }
  }

  // 根据单位计算字节数
  if (unit.empty() || unit == "B") {
    return static_cast<size_t>(value);
  } else if (unit == "K" || unit == "KB") {
    return static_cast<size_t>(value * 1024);
  } else if (unit == "M" || unit == "MB") {
    return static_cast<size_t>(value * 1024 * 1024);
  } else if (unit == "G" || unit == "GB") {
    return static_cast<size_t>(value * 1024 * 1024 * 1024);
  }

  // 未知单位，尝试作为字节数
  return static_cast<size_t>(value);
}

}  // namespace ksana_llm
