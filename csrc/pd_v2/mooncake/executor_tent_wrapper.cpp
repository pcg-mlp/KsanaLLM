/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/executor_tent_wrapper.h"

#include <cstdlib>

#include <chrono>
#include <thread>

#include "common/base/status.h"
#include "fmt/core.h"
#include "transfer_engine.h"
#include "transport/transport.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

// Fallback when the caller passes 0 — keeps single-engine + single-
// executor dev launches working without explicit config. Multi-rank
// deploys must override (see inference_executor.cpp's per-rank
// assignment via PdV2ConnectorConfig::ExecutorConfig::rpc_port).
constexpr uint64_t kExecutorRpcPortFallback = 12345;

constexpr int kSubmitPollIntervalUs = 200;
constexpr int kSubmitPollMaxIters = 5000;  // 1s ceiling on poll

}  // namespace

ExecutorTentWrapper::ExecutorTentWrapper() = default;

ExecutorTentWrapper::~ExecutorTentWrapper() {
  if (initialized_) {
    Shutdown().GetMessage();
  }
}

Status ExecutorTentWrapper::Initialize(const std::string& metadata_addr, const std::string& local_server_name,
                                       uint64_t rpc_port) {
  if (initialized_) {
    return Status(RET_INVALID_ARGUMENT, "ExecutorTentWrapper already initialized");
  }
  local_server_name_ = local_server_name;
  const uint64_t effective_port = rpc_port == 0 ? kExecutorRpcPortFallback : rpc_port;

  // Force tent backend. See engine_tent_wrapper.cpp for the rationale —
  // pd_v2 always uses tent, no caller override.
  setenv("MC_USE_TENT", "1", /*overwrite=*/1);

  // MC_TENT_CONF (rpc_server_hostname / topology / transports) is the
  // upstream Mooncake config knob. We ship no default (HCA names + GPU
  // topology are fleet-specific); operator must export MC_TENT_CONF
  // before launch.
  if (getenv("MC_TENT_CONF") == nullptr) {
    KLLM_LOG_WARNING << "ExecutorTentWrapper: MC_TENT_CONF env not set — tent "
                     << "may auto-pick HCAs that don't share a NUMA/PCI domain "
                     << "with the GPU buffer being registered, leading to "
                     << "ibv_reg_mr EFAULT failures.";
  }

  KLLM_LOG_INFO << "ExecutorTentWrapper: before make_unique<TransferEngine> port=" << effective_port;
  try {
    engine_ = std::make_unique<mooncake::TransferEngine>(/*auto_discover=*/false);
  } catch (const std::exception& e) {
    KLLM_LOG_ERROR << "ExecutorTentWrapper: make_unique<TransferEngine> threw: " << e.what();
    return Status(RET_RUNTIME_FAILED, fmt::format("TransferEngine ctor threw: {}", e.what()));
  }
  const std::string md = metadata_addr.empty() ? "P2PHANDSHAKE" : metadata_addr;
  KLLM_LOG_INFO << "ExecutorTentWrapper: before engine_->init md=" << md << " name=" << local_server_name
                << " port=" << effective_port;
  int init_rc = -1;
  try {
    init_rc = engine_->init(md, local_server_name, /*ip_or_host_name=*/"", effective_port);
  } catch (const std::exception& e) {
    KLLM_LOG_ERROR << "ExecutorTentWrapper: engine_->init threw: " << e.what();
    return Status(RET_RUNTIME_FAILED, fmt::format("engine init threw: {}", e.what()));
  }
  KLLM_LOG_INFO << "ExecutorTentWrapper: engine_->init returned rc=" << init_rc;
  if (init_rc != 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("mooncake TransferEngine::init failed metadata={}", md));
  }
  // Under tent backend, transports are configured through MC_TENT_CONF
  // (RDMA + TCP both enabled). installTransport intentionally returns
  // nullptr in tent mode — not an error.
  void* args = nullptr;
  (void)engine_->installTransport("rdma", &args);
  (void)engine_->installTransport("tcp", &args);

  KLLM_LOG_INFO << "ExecutorTentWrapper::Initialize metadata=" << md << " local_server_name=" << local_server_name
                << " rpc_port=" << effective_port << (rpc_port == 0 ? " (fallback default)" : " (from config)");
  initialized_ = true;
  return Status();
}

Status ExecutorTentWrapper::Shutdown() {
  if (!initialized_) return Status();
  KLLM_LOG_INFO << "ExecutorTentWrapper::Shutdown local_server_name=" << local_server_name_;
  engine_.reset();
  initialized_ = false;
  return Status();
}

Status ExecutorTentWrapper::RegisterKvPool(void* base, size_t total_bytes, const std::string& location) {
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "ExecutorTentWrapper not initialized");
  }
  if (engine_->registerLocalMemory(base, total_bytes, location,
                                   /*remote_accessible=*/true,
                                   /*update_metadata=*/true) != 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("mooncake registerLocalMemory failed base={} bytes={} loc={}", base,
                                                  total_bytes, location));
  }
  KLLM_LOG_INFO << "ExecutorTentWrapper::RegisterKvPool base=" << base << " bytes=" << total_bytes
                << " location=" << location;
  return Status();
}

Status ExecutorTentWrapper::SubmitWrite(const std::string& target_server_name, uint64_t target_va, void* source,
                                        size_t length) {
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "ExecutorTentWrapper not initialized");
  }
  // openSegment resolves a SegmentID for the remote engine's registered
  // memory region. The same SegmentID covers all RDMA-accessible memory
  // that engine registered via registerLocalMemory; target_offset is the
  // absolute remote VA, not relative to any sub-region.
  mooncake::SegmentHandle seg = engine_->openSegment(target_server_name);
  if (seg == mooncake::Transport::SegmentID(-1) || static_cast<int64_t>(seg) < 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("openSegment({}) failed", target_server_name));
  }

  mooncake::BatchID batch = engine_->allocateBatchID(/*batch_size=*/1);
  if (batch == mooncake::INVALID_BATCH_ID) {
    return Status(RET_RUNTIME_FAILED, "allocateBatchID failed");
  }

  mooncake::TransferRequest req;
  req.opcode = mooncake::TransferRequest::WRITE;
  req.source = source;
  req.target_id = static_cast<mooncake::SegmentID>(seg);
  req.target_offset = target_va;
  req.length = length;

  mooncake::Status s = engine_->submitTransfer(batch, {req});
  if (!s.ok()) {
    engine_->freeBatchID(batch);
    return Status(RET_RUNTIME_FAILED, fmt::format("submitTransfer failed code={} msg={}", static_cast<int>(s.code()),
                                                  std::string(s.message())));
  }

  // Poll until COMPLETED / FAILED / TIMEOUT. Mooncake CQEs are reaped by
  // a background poll thread inside the transport, so this is just a
  // status check; the transport handles the actual completion event.
  for (int i = 0; i < kSubmitPollMaxIters; ++i) {
    mooncake::TransferStatus ts;
    mooncake::Status gs = engine_->getBatchTransferStatus(batch, ts);
    if (!gs.ok()) {
      engine_->freeBatchID(batch);
      return Status(RET_RUNTIME_FAILED,
                    fmt::format("getBatchTransferStatus failed code={}", static_cast<int>(gs.code())));
    }
    if (ts.s == mooncake::Transport::COMPLETED) {
      engine_->freeBatchID(batch);
      return Status();
    }
    if (ts.s == mooncake::Transport::FAILED || ts.s == mooncake::Transport::TIMEOUT ||
        ts.s == mooncake::Transport::CANCELED || ts.s == mooncake::Transport::INVALID) {
      engine_->freeBatchID(batch);
      return Status(RET_RUNTIME_FAILED, fmt::format("submitTransfer terminal state={}", static_cast<int>(ts.s)));
    }
    std::this_thread::sleep_for(std::chrono::microseconds(kSubmitPollIntervalUs));
  }
  engine_->freeBatchID(batch);
  return Status(RET_RUNTIME_FAILED, "submitTransfer poll timeout (>1s)");
}

Status ExecutorTentWrapper::FreeBatchID(uint64_t batch_id) {
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "ExecutorTentWrapper not initialized");
  }
  // mooncake::TransferEngine::freeBatchID returns mooncake::Status; we
  // don't surface its detailed error code here — cancel is best-effort.
  mooncake::Status s = engine_->freeBatchID(static_cast<mooncake::BatchID>(batch_id));
  if (!s.ok()) {
    KLLM_LOG_WARNING << "ExecutorTentWrapper::FreeBatchID(" << batch_id
                     << ") failed code=" << static_cast<int>(s.code());
    return Status(RET_RUNTIME_FAILED, "freeBatchID failed");
  }
  return Status();
}

Status ExecutorTentWrapper::SubmitWriteAsync(const std::string& target_server_name, uint64_t target_va, void* source,
                                             size_t length, uint64_t* out_batch_id) {
  if (out_batch_id) *out_batch_id = 0;
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "ExecutorTentWrapper not initialized");
  }
  mooncake::SegmentHandle seg = engine_->openSegment(target_server_name);
  if (seg == mooncake::Transport::SegmentID(-1) || static_cast<int64_t>(seg) < 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("openSegment({}) failed", target_server_name));
  }
  mooncake::BatchID batch = engine_->allocateBatchID(/*batch_size=*/1);
  if (batch == mooncake::INVALID_BATCH_ID) {
    return Status(RET_RUNTIME_FAILED, "allocateBatchID failed");
  }
  mooncake::TransferRequest req;
  req.opcode = mooncake::TransferRequest::WRITE;
  req.source = source;
  req.target_id = static_cast<mooncake::SegmentID>(seg);
  req.target_offset = target_va;
  req.length = length;
  mooncake::Status s = engine_->submitTransfer(batch, {req});
  if (!s.ok()) {
    engine_->freeBatchID(batch);
    return Status(RET_RUNTIME_FAILED, fmt::format("submitTransfer failed code={}", static_cast<int>(s.code())));
  }
  if (out_batch_id) *out_batch_id = static_cast<uint64_t>(batch);
  return Status();
}

Status ExecutorTentWrapper::SubmitWriteBatchAsync(const std::string& target_server_name,
                                                  const std::vector<BatchWriteEntry>& entries, uint64_t* out_batch_id) {
  if (out_batch_id) *out_batch_id = 0;
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "ExecutorTentWrapper not initialized");
  }
  if (entries.empty()) {
    return Status(RET_INVALID_ARGUMENT, "SubmitWriteBatchAsync called with empty entries");
  }
  // Resolve the remote segment once — every entry targets the same peer.
  mooncake::SegmentHandle seg = engine_->openSegment(target_server_name);
  if (seg == mooncake::Transport::SegmentID(-1) || static_cast<int64_t>(seg) < 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("openSegment({}) failed", target_server_name));
  }
  mooncake::BatchID batch = engine_->allocateBatchID(/*batch_size=*/entries.size());
  if (batch == mooncake::INVALID_BATCH_ID) {
    return Status(RET_RUNTIME_FAILED, "allocateBatchID failed");
  }
  std::vector<mooncake::TransferRequest> reqs;
  reqs.reserve(entries.size());
  for (const auto& e : entries) {
    mooncake::TransferRequest req;
    req.opcode = mooncake::TransferRequest::WRITE;
    req.source = e.source;
    req.target_id = static_cast<mooncake::SegmentID>(seg);
    req.target_offset = e.target_va;
    req.length = e.length;
    reqs.push_back(req);
  }
  mooncake::Status s = engine_->submitTransfer(batch, reqs);
  if (!s.ok()) {
    engine_->freeBatchID(batch);
    return Status(RET_RUNTIME_FAILED,
                  fmt::format("submitTransfer(batch_size={}) failed code={}", reqs.size(), static_cast<int>(s.code())));
  }
  if (out_batch_id) *out_batch_id = static_cast<uint64_t>(batch);
  return Status();
}

Status ExecutorTentWrapper::GetBatchStatus(uint64_t batch_id, bool* out_completed, bool* out_failed) {
  if (out_completed) *out_completed = false;
  if (out_failed) *out_failed = false;
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "ExecutorTentWrapper not initialized");
  }
  mooncake::TransferStatus ts;
  mooncake::Status gs = engine_->getBatchTransferStatus(static_cast<mooncake::BatchID>(batch_id), ts);
  if (!gs.ok()) {
    return Status(RET_RUNTIME_FAILED,
                  fmt::format("getBatchTransferStatus failed code={}", static_cast<int>(gs.code())));
  }
  if (ts.s == mooncake::Transport::COMPLETED) {
    if (out_completed) *out_completed = true;
  } else if (ts.s == mooncake::Transport::FAILED || ts.s == mooncake::Transport::TIMEOUT ||
             ts.s == mooncake::Transport::CANCELED || ts.s == mooncake::Transport::INVALID) {
    if (out_failed) *out_failed = true;
  }
  return Status();
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
