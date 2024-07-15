/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The writer of the profiler.
class ProfileWriter {
 public:
  ProfileWriter();
  ~ProfileWriter();

  // Write message to disk file.
  Status Write(const std::string& message);
  Status Write(const std::vector<std::string>& messages);

 private:
  // The file name of current profiler.
  std::string profiler_file_;

  // The output file stream.
  std::ofstream profiler_stream_;
};

}  // namespace ksana_llm
