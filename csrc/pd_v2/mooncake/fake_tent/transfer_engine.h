/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// FAKE transfer_engine.h — a self-contained, in-process stand-in for
// mooncake::TransferEngine. Declares exactly the surface the pd_v2
// wrappers/connectors call (init / installTransport / openSegment /
// closeSegment / registerLocalMemory / allocateBatchID / submitTransfer /
// getBatchTransferStatus / freeBatchID / getNotifies / sendNotifyByID) plus
// the mooncake::* type aliases and INVALID_BATCH_ID from the real header.
//
// Behaviour (see fake_tent_engine.cpp): all engine instances share a
// process-global registry. openSegment(name) resolves a peer by its
// init() server_name; sendNotifyByID routes a NotifyDesc into the peer's
// in-process queue; submitTransfer performs a real in-process memcpy
// (single address space ⇒ the remote VA target_offset is a valid local
// pointer) and reports COMPLETED. This lets the REAL connector/wrapper
// .cpp execute end-to-end with no RDMA, no metadata server, no second
// process — so their lines are exercised for coverage.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/base/status.h"
#include "transfer_metadata.h"
#include "transport/transport.h"

namespace mooncake {

using TransferRequest = Transport::TransferRequest;
using TransferStatus = Transport::TransferStatus;
using TransferStatusEnum = Transport::TransferStatusEnum;
using SegmentHandle = Transport::SegmentHandle;
using SegmentID = Transport::SegmentID;
using BatchID = Transport::BatchID;
const static BatchID INVALID_BATCH_ID = UINT64_MAX;
using BufferEntry = Transport::BufferEntry;

class TransferEngine {
 public:
  explicit TransferEngine(bool auto_discover = false);
  ~TransferEngine();

  TransferEngine(const TransferEngine&) = delete;
  TransferEngine& operator=(const TransferEngine&) = delete;

  int init(const std::string& metadata_conn_string, const std::string& local_server_name,
           const std::string& ip_or_host_name = "", uint64_t rpc_port = 12345);

  int freeEngine();

  Transport* installTransport(const std::string& proto, void** args);

  SegmentHandle openSegment(const std::string& segment_name);

  int closeSegment(SegmentHandle handle);

  int registerLocalMemory(void* addr, size_t length, const std::string& location = "*",
                          bool remote_accessible = true, bool update_metadata = true);

  int unregisterLocalMemory(void* addr, bool update_metadata = true);

  BatchID allocateBatchID(size_t batch_size);

  Status freeBatchID(BatchID batch_id);

  Status submitTransfer(BatchID batch_id, const std::vector<TransferRequest>& entries);

  Status getBatchTransferStatus(BatchID batch_id, TransferStatus& status);

  int getNotifies(std::vector<TransferMetadata::NotifyDesc>& notifies);

  int sendNotifyByID(SegmentID target_id, TransferMetadata::NotifyDesc notify_msg);

  int sendNotifyByName(std::string remote_agent, TransferMetadata::NotifyDesc notify_msg);

  // Fake-only knobs for tests (NOT in the real header). Used by the
  // coverage test to force failure branches in the connector code.
  static void FailNextSubmitTransfer();
  static void Reset();

 private:
  uint64_t self_id_ = 0;
  std::string name_;
};

}  // namespace mooncake
