/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// FAKE transport/transport.h — minimal stand-in providing the nested
// types the pd_v2 mooncake wrappers reference via mooncake::Transport::*
// and the mooncake::{TransferRequest,TransferStatus,SegmentID,...} aliases.
#pragma once

#include <cstddef>
#include <cstdint>

namespace mooncake {

class Transport {
 public:
  using SegmentID = uint64_t;
  using SegmentHandle = SegmentID;
  using BatchID = uint64_t;

  struct BufferEntry {
    void* addr = nullptr;
    size_t length = 0;
  };

  struct TransferRequest {
    enum OpCode { READ, WRITE };
    OpCode opcode = READ;
    void* source = nullptr;
    SegmentID target_id = 0;
    uint64_t target_offset = 0;
    size_t length = 0;
  };

  // Order/values mirror the real upstream enum so any code comparing
  // against specific enumerators behaves identically.
  enum TransferStatusEnum {
    WAITING,
    PENDING,
    INVALID,
    CANCELED,
    COMPLETED,
    TIMEOUT,
    FAILED,
  };

  struct TransferStatus {
    TransferStatusEnum s = WAITING;
    size_t transferred_bytes = 0;
  };

  virtual ~Transport() = default;
};

}  // namespace mooncake
