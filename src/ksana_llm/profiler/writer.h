/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <string>

#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The writer of the profiler.
class ProfileWriter {
 public:
  ProfileWriter();

  // Write message to disk file.
  Status Write(const std::string& message);

 private:
  // The file name of current profiler.
  std::string profiler_file_;
};

}  // namespace ksana_llm
