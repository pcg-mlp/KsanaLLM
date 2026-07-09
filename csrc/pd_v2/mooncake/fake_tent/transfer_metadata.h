/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// FAKE transfer_metadata.h — provides mooncake::TransferMetadata::NotifyDesc,
// the only TransferMetadata symbol the pd_v2 wrappers reference.
#pragma once

#include <string>
#include <vector>

namespace mooncake {

class TransferMetadata {
 public:
  struct NotifyDesc {
    std::string name;
    std::string notify_msg;
  };
};

}  // namespace mooncake
