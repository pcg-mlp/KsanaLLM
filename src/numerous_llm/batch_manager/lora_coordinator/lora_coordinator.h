/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

class LoraCoordinator {
public:
  explicit LoraCoordinator(
      const LoraCoordinatorConfig &lora_coordinator_config);
};

} // namespace numerous_llm
