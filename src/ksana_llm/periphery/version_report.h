
/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

struct VersionReportOption {};

namespace ksana_llm {
class VersionReport {
 public:
  // Singleton instance getter
  static VersionReport& GetInstance() {
    static VersionReport instance;
    return instance;
  }
  // Initialize version reporting with given options
  void InitVersionReport(const VersionReportOption& option = VersionReportOption());
};
}  //  namespace ksana_llm
