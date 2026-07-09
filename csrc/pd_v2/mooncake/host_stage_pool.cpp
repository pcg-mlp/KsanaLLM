/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/host_stage_pool.h"

#include <sys/mman.h>

#include <cerrno>
#include <cstring>

#include "fmt/core.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

HostStagePool::HostStagePool() = default;

HostStagePool::~HostStagePool() {
  if (initialized_) {
    Shutdown().GetMessage();  // best-effort
  }
}

Status HostStagePool::Init(size_t total_bytes, size_t max_peers) {
  std::lock_guard<std::mutex> lg(mu_);
  if (initialized_) {
    return Status(RET_INVALID_ARGUMENT, "HostStagePool already initialized");
  }
  if (total_bytes == 0 || total_bytes % kSlotSize != 0) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("HostStagePool total_bytes={} must be non-zero multiple of slot {}",
                                                    total_bytes, kSlotSize));
  }
  if (max_peers == 0) {
    return Status(RET_INVALID_ARGUMENT, "HostStagePool max_peers must be > 0");
  }
  const size_t total_slots = total_bytes / kSlotSize;
  if (total_slots < max_peers) {
    return Status(RET_INVALID_ARGUMENT,
                  fmt::format("HostStagePool total_slots={} < max_peers={}; pool too small", total_slots, max_peers));
  }

  // MAP_POPULATE pre-faults the pages so registerLocalMemory's ibv_reg_mr
  // doesn't trigger a flood of minor page-faults during MR setup.
  void* base = ::mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                      /*fd=*/-1, /*offset=*/0);
  if (base == MAP_FAILED) {
    return Status(RET_RUNTIME_FAILED, fmt::format("HostStagePool mmap({} bytes) failed errno={}", total_bytes, errno));
  }

  // Pin so the kernel can't swap pages out from under ibv_reg_mr's MR.
  // Best-effort: ibv_reg_mr also pins, so a failure here (typically
  // RLIMIT_MEMLOCK too low on dev hosts) is logged and ignored. Real RDMA
  // hosts run with `ulimit -l unlimited`.
  if (::mlock(base, total_bytes) != 0) {
    KLLM_LOG_WARNING << "HostStagePool mlock(" << total_bytes << ") failed errno=" << errno
                     << " — proceeding (raise RLIMIT_MEMLOCK on prod hosts)";
  }

  base_ = base;
  total_slots_ = total_slots;
  max_peers_ = max_peers;
  // Static partition: integer-divide and accept the tail loss. The unused
  // tail slots beyond max_peers * slots_per_peer never get assigned —
  // that's the cost of a stable peer→slot map across peer churn.
  slots_per_peer_ = total_slots / max_peers;
  initialized_ = true;
  KLLM_LOG_INFO << "HostStagePool::Init bytes=" << total_bytes << " slots=" << total_slots << " max_peers=" << max_peers
                << " slots_per_peer=" << slots_per_peer_ << " base=" << base_;
  return Status();
}

Status HostStagePool::Shutdown() {
  std::lock_guard<std::mutex> lg(mu_);
  if (!initialized_) return Status();
  // munmap implicitly drops mlock pins; no separate munlock needed.
  if (::munmap(base_, total_slots_ * kSlotSize) != 0) {
    KLLM_LOG_WARNING << "HostStagePool munmap errno=" << errno;
  }
  base_ = nullptr;
  total_slots_ = 0;
  slots_per_peer_ = 0;
  max_peers_ = 0;
  next_partition_index_ = 0;
  peer_map_.clear();
  initialized_ = false;
  return Status();
}

Status HostStagePool::RegisterPeer(const std::string& decode_addr, PeerPartition* out) {
  if (out == nullptr) return Status(RET_INVALID_ARGUMENT, "out is null");
  std::lock_guard<std::mutex> lg(mu_);
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "HostStagePool not initialized");
  }
  auto it = peer_map_.find(decode_addr);
  if (it != peer_map_.end()) {
    *out = it->second;
    return Status();
  }
  // TODO(shawnding): 当前用单调递增的 next_partition_index_ 分配 partition，从不回收。
  // 同一 decode_addr 重启会复用旧 partition（上面的幂等分支），不会泄漏；但若
  // Decode 永久下线或地址发生变化（例如重新调度后 IP 改变），其 partition 不会
  // 归还，累计出现 > max_peers 个不同地址后会在此耗尽。后续应配合 UnregisterPeer
  // + free-list 回收已下线 peer 的槽位（触发点见 mooncake_engine_connector.cpp
  // 中清理 decode_epoch_cache_ 的逻辑）。
  if (next_partition_index_ >= max_peers_) {
    return Status(RET_RUNTIME_FAILED,
                  fmt::format("HostStagePool out of peer partitions max_peers={} addr={}", max_peers_, decode_addr));
  }
  PeerPartition part;
  part.base_slot = next_partition_index_ * slots_per_peer_;
  part.n_slots = slots_per_peer_;
  ++next_partition_index_;
  peer_map_[decode_addr] = part;
  *out = part;
  KLLM_LOG_INFO << "HostStagePool::RegisterPeer addr=" << decode_addr << " base_slot=" << part.base_slot
                << " n_slots=" << part.n_slots;
  return Status();
}

Status HostStagePool::PeerPartitionFor(const std::string& decode_addr, PeerPartition* out) const {
  if (out == nullptr) return Status(RET_INVALID_ARGUMENT, "out is null");
  std::lock_guard<std::mutex> lg(mu_);
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "HostStagePool not initialized");
  }
  auto it = peer_map_.find(decode_addr);
  if (it == peer_map_.end()) {
    return Status(RET_RUNTIME_FAILED, fmt::format("HostStagePool peer not registered addr={}", decode_addr));
  }
  *out = it->second;
  return Status();
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
