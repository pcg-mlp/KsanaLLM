/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Implementation of the FAKE mooncake::TransferEngine. Process-global
// registry: engines register under their init() server_name; peers resolve
// each other via openSegment; notifies are routed in-process; transfers are
// real local memcpy (single address space). No RDMA / metadata server.

#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>
#include "transfer_engine.h"

namespace mooncake {
namespace {

struct EngineNode {
  std::string name;
  std::mutex mu;
  std::deque<TransferMetadata::NotifyDesc> notifies;
};

struct Registry {
  std::mutex mu;
  std::unordered_map<uint64_t, EngineNode*> by_id;
  std::unordered_map<std::string, uint64_t> name_to_id;
  std::unordered_map<uint64_t, TransferStatusEnum> batch_status;
  uint64_t next_engine_id = 1;
  uint64_t next_batch_id = 1;
  bool fail_next_submit = false;
};

Registry& Reg() {
  static Registry r;
  return r;
}

}  // namespace

TransferEngine::TransferEngine(bool /*auto_discover*/) {}

TransferEngine::~TransferEngine() { freeEngine(); }

int TransferEngine::init(const std::string& /*metadata*/, const std::string& local_server_name,
                         const std::string& /*ip*/, uint64_t /*rpc_port*/) {
  auto& r = Reg();
  std::lock_guard<std::mutex> lk(r.mu);
  self_id_ = r.next_engine_id++;
  name_ = local_server_name;
  auto* node = new EngineNode();
  node->name = local_server_name;
  r.by_id[self_id_] = node;
  r.name_to_id[local_server_name] = self_id_;
  return 0;
}

int TransferEngine::freeEngine() {
  if (self_id_ == 0) return 0;
  auto& r = Reg();
  std::lock_guard<std::mutex> lk(r.mu);
  auto it = r.by_id.find(self_id_);
  if (it != r.by_id.end()) {
    delete it->second;
    r.by_id.erase(it);
  }
  r.name_to_id.erase(name_);
  self_id_ = 0;
  return 0;
}

Transport* TransferEngine::installTransport(const std::string& /*proto*/, void** /*args*/) {
  // Tent-mode returns nullptr; wrappers treat that as success.
  return nullptr;
}

SegmentHandle TransferEngine::openSegment(const std::string& segment_name) {
  auto& r = Reg();
  std::lock_guard<std::mutex> lk(r.mu);
  auto it = r.name_to_id.find(segment_name);
  if (it == r.name_to_id.end()) {
    return static_cast<SegmentHandle>(-1);  // wrapper checks (int64_t)handle < 0
  }
  return static_cast<SegmentHandle>(it->second);
}

int TransferEngine::closeSegment(SegmentHandle /*handle*/) { return 0; }

int TransferEngine::registerLocalMemory(void* /*addr*/, size_t /*length*/, const std::string& /*location*/,
                                        bool /*remote_accessible*/, bool /*update_metadata*/) {
  return 0;
}

int TransferEngine::unregisterLocalMemory(void* /*addr*/, bool /*update_metadata*/) { return 0; }

BatchID TransferEngine::allocateBatchID(size_t /*batch_size*/) {
  auto& r = Reg();
  std::lock_guard<std::mutex> lk(r.mu);
  BatchID id = r.next_batch_id++;
  r.batch_status[id] = Transport::WAITING;
  return id;
}

Status TransferEngine::freeBatchID(BatchID batch_id) {
  auto& r = Reg();
  std::lock_guard<std::mutex> lk(r.mu);
  r.batch_status.erase(batch_id);
  return Status::OK();
}

Status TransferEngine::submitTransfer(BatchID batch_id, const std::vector<TransferRequest>& entries) {
  auto& r = Reg();
  {
    std::lock_guard<std::mutex> lk(r.mu);
    if (r.fail_next_submit) {
      r.fail_next_submit = false;
      r.batch_status[batch_id] = Transport::FAILED;
      return Status(Status::Code::kRuntime, "fake submitTransfer forced failure");
    }
  }
  // Single address space: target_offset is a valid local pointer (the peer
  // registered the same VA). Perform the WRITE so tests can verify payload.
  for (const auto& e : entries) {
    if (e.opcode == TransferRequest::WRITE && e.source != nullptr && e.target_offset != 0 && e.length != 0) {
      std::memcpy(reinterpret_cast<void*>(e.target_offset), e.source, e.length);
    }
  }
  {
    std::lock_guard<std::mutex> lk(r.mu);
    r.batch_status[batch_id] = Transport::COMPLETED;
  }
  return Status::OK();
}

Status TransferEngine::getBatchTransferStatus(BatchID batch_id, TransferStatus& status) {
  auto& r = Reg();
  std::lock_guard<std::mutex> lk(r.mu);
  auto it = r.batch_status.find(batch_id);
  status.s = (it == r.batch_status.end()) ? Transport::COMPLETED : it->second;
  return Status::OK();
}

int TransferEngine::getNotifies(std::vector<TransferMetadata::NotifyDesc>& notifies) {
  notifies.clear();
  auto& r = Reg();
  EngineNode* node = nullptr;
  {
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.by_id.find(self_id_);
    if (it == r.by_id.end()) return 0;
    node = it->second;
  }
  std::lock_guard<std::mutex> lk(node->mu);
  while (!node->notifies.empty()) {
    notifies.push_back(std::move(node->notifies.front()));
    node->notifies.pop_front();
  }
  return 0;
}

int TransferEngine::sendNotifyByID(SegmentID target_id, TransferMetadata::NotifyDesc notify_msg) {
  auto& r = Reg();
  EngineNode* node = nullptr;
  {
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.by_id.find(static_cast<uint64_t>(target_id));
    if (it == r.by_id.end()) return -1;
    node = it->second;
  }
  std::lock_guard<std::mutex> lk(node->mu);
  node->notifies.push_back(std::move(notify_msg));
  return 0;
}

int TransferEngine::sendNotifyByName(std::string remote_agent, TransferMetadata::NotifyDesc notify_msg) {
  return sendNotifyByID(openSegment(remote_agent), std::move(notify_msg));
}

void TransferEngine::FailNextSubmitTransfer() {
  auto& r = Reg();
  std::lock_guard<std::mutex> lk(r.mu);
  r.fail_next_submit = true;
}

void TransferEngine::Reset() {
  auto& r = Reg();
  std::lock_guard<std::mutex> lk(r.mu);
  for (auto& kv : r.by_id) delete kv.second;
  r.by_id.clear();
  r.name_to_id.clear();
  r.batch_status.clear();
  r.next_engine_id = 1;
  r.next_batch_id = 1;
  r.fail_next_submit = false;
}

}  // namespace mooncake
