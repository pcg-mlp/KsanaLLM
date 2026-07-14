/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace ksana_llm {

class ServiceLifetimeManagerInterface {
 public:
  virtual ~ServiceLifetimeManagerInterface() {}

  // Stop the ksana service.
  virtual void ShutdownService() = 0;
};

}  // namespace ksana_llm
