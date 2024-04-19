/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

class LoraCoordinator {
 public:
  explicit LoraCoordinator(const LoraCoordinatorConfig &lora_coordinator_config);
};

}  // namespace ksana_llm
