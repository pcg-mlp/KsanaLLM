/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "pd_v2/public/pd_v2_connector.h"
#include "pd_v2/public/pd_v2_types.h"

namespace ksana_llm {

class Request;

namespace pd_v2 {

// Tracks per-request barrier + sampler completion and sends
// SendPrefillComplete when both gates have fired.
//
// 生命周期：PD-V2 prefill 的 Request 没有 Python/StreamingIterator 兜底，其完整生命周期
// 由本类全程持有 —— Admit() 时把 shared_ptr<Request> 存入 alive 表，直到 barrier 与 sampler
// 两个 gate 都就绪后，在 TryCompleteUnlocked() 中连同发送一起 erase 释放。因此 InferRequest
// 不再需要额外 pin 住 Request（原 SetRequestOwner 已删除）。
//
// 释放时机的安全性：tokens_ready 由 request 的 waiter 回调触发，而 waiter->Notify 经
// 调度器 StopRequest 的 HasInflightTask 延迟，恒在 in-flight 执行任务（含 ModelDriver
// 的采样/IPC drain，如 UpdateSamplingOutput 写 logprobs）全部 drain 之后才触发；barrier
// gate 在 forward 末层即上报，必早于 tokens_ready。两 gate 取较晚者，故 erase（释放
// Request）恒晚于最后一次对 InferRequest 引用成员的访问，无 UAF。
class PdV2PrefillCompletion {
 public:
  struct Deps {
    std::shared_ptr<PdV2ConnectorInterface> connector = nullptr;
    std::function<PeerHealth()> snapshot_fn = {};
  };

  explicit PdV2PrefillCompletion(Deps deps);

  std::function<void()> MakeRequestFinishedCallback(uint64_t req_id) const;

  void Admit(uint64_t req_id, const std::shared_ptr<Request>& request, const std::string& peer_addr,
             uint32_t expected_barrier_reports);

  void OnBarrierDone(int64_t req_id, uint8_t status);

  size_t NumAdmittedForTest() const;
  size_t NumCompletedForTest() const;
  size_t NumAliveForTest() const;

 private:
  struct PendingEntry {
    std::shared_ptr<Request> request;
    std::string peer_addr;
    bool tokens_ready = false;
    PdCompleteStatus token_status = PdCompleteStatus::kOk;
    std::vector<int> tokens;
    uint32_t expected_barrier_reports = 0;
    uint32_t received_barrier_reports = 0;
    bool barrier_done = false;
    PdCompleteStatus barrier_status = PdCompleteStatus::kOk;
  };

  struct SharedState {
    mutable std::mutex mu;
    std::unordered_map<uint64_t, PendingEntry> alive;
    size_t num_admitted = 0;
    size_t num_completed = 0;
  };

  static void OnRequestFinished(std::weak_ptr<SharedState> state_weak,
                                std::shared_ptr<PdV2ConnectorInterface> connector,
                                std::function<PeerHealth()> snapshot_fn, uint64_t req_id);

  static bool TryCompleteUnlocked(SharedState* state, uint64_t req_id, std::shared_ptr<Request>* out_request,
                                  std::string* out_peer_addr, PdCompleteStatus* out_status,
                                  std::vector<int>* out_tokens);

  static void DoSendPrefillComplete(std::shared_ptr<PdV2ConnectorInterface> connector,
                                    const std::function<PeerHealth()>& snapshot_fn,
                                    const std::shared_ptr<Request>& request, const std::string& peer_addr,
                                    PdCompleteStatus status, std::vector<int> tokens, uint64_t req_id);

  Deps deps_;
  std::shared_ptr<SharedState> state_;
};

}  // namespace pd_v2
}  // namespace ksana_llm
