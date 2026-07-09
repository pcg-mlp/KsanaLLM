/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "endpoints/pd_v2/pd_v2_prefill_completion.h"

#include "profiler/timer.h"
#include "utils/logger.h"
#include "utils/request.h"
#include "utils/status.h"
#include "utils/waiter.h"

namespace ksana_llm {
namespace pd_v2 {

PdV2PrefillCompletion::PdV2PrefillCompletion(Deps deps)
    : deps_(std::move(deps)), state_(std::make_shared<SharedState>()) {}

std::function<void()> PdV2PrefillCompletion::MakeRequestFinishedCallback(uint64_t req_id) const {
  std::weak_ptr<SharedState> state_weak = state_;
  std::shared_ptr<PdV2ConnectorInterface> connector = deps_.connector;
  std::function<PeerHealth()> snapshot_fn = deps_.snapshot_fn;
  return
      [state_weak, connector, snapshot_fn, req_id]() { OnRequestFinished(state_weak, connector, snapshot_fn, req_id); };
}

void PdV2PrefillCompletion::Admit(uint64_t req_id, const std::shared_ptr<Request>& request,
                                  const std::string& peer_addr, uint32_t expected_barrier_reports) {
  std::lock_guard<std::mutex> lk(state_->mu);
  PendingEntry entry;
  entry.request = request;
  entry.peer_addr = peer_addr;
  entry.expected_barrier_reports = expected_barrier_reports;
  state_->alive[req_id] = std::move(entry);
  ++state_->num_admitted;
}

void PdV2PrefillCompletion::OnBarrierDone(int64_t req_id, uint8_t status) {
  std::shared_ptr<Request> request;
  std::string peer_addr;
  PdCompleteStatus final_status = PdCompleteStatus::kOk;
  std::vector<int> final_tokens;
  bool ready = false;
  {
    std::lock_guard<std::mutex> lk(state_->mu);
    auto it = state_->alive.find(static_cast<uint64_t>(req_id));
    if (it == state_->alive.end()) {
      KLLM_LOG_WARNING << "PdV2PrefillCompletion::OnBarrierDone: req_id " << req_id
                       << " not in alive (already completed or never admitted)";
      return;
    }
    auto& entry = it->second;
    ++entry.received_barrier_reports;
    if (status != 0) {
      entry.barrier_status = PdCompleteStatus::kFailed;
    }
    KLLM_LOG_INFO << "PdV2PrefillCompletion::OnBarrierDone: req_id=" << req_id
                  << " received=" << entry.received_barrier_reports << " expected=" << entry.expected_barrier_reports
                  << " incoming_status=" << static_cast<int>(status)
                  << " aggregated_status=" << static_cast<int>(entry.barrier_status);
    if (entry.received_barrier_reports >= entry.expected_barrier_reports) {
      entry.barrier_done = true;
      ready = TryCompleteUnlocked(state_.get(), static_cast<uint64_t>(req_id), &request, &peer_addr, &final_status,
                                  &final_tokens);
    }
  }
  if (ready) {
    DoSendPrefillComplete(deps_.connector, deps_.snapshot_fn, request, peer_addr, final_status, std::move(final_tokens),
                          static_cast<uint64_t>(req_id));
  }
}

bool PdV2PrefillCompletion::TryCompleteUnlocked(SharedState* state, uint64_t req_id,
                                                std::shared_ptr<Request>* out_request, std::string* out_peer_addr,
                                                PdCompleteStatus* out_status, std::vector<int>* out_tokens) {
  auto it = state->alive.find(req_id);
  if (it == state->alive.end()) return false;
  auto& entry = it->second;
  if (!entry.tokens_ready || !entry.barrier_done) return false;
  *out_request = entry.request;
  *out_peer_addr = entry.peer_addr;
  if (entry.token_status != PdCompleteStatus::kOk || entry.barrier_status != PdCompleteStatus::kOk) {
    *out_status = PdCompleteStatus::kFailed;
  } else {
    *out_status = PdCompleteStatus::kOk;
  }
  *out_tokens = std::move(entry.tokens);
  state->alive.erase(it);
  ++state->num_completed;
  return true;
}

void PdV2PrefillCompletion::DoSendPrefillComplete(std::shared_ptr<PdV2ConnectorInterface> connector,
                                                  const std::function<PeerHealth()>& snapshot_fn,
                                                  const std::shared_ptr<Request>& request, const std::string& peer_addr,
                                                  PdCompleteStatus status, std::vector<int> tokens, uint64_t req_id) {
  if (request != nullptr && tokens.empty() && status == PdCompleteStatus::kOk) {
    std::lock_guard<std::mutex> ol(request->output_mutex);
    tokens = request->output_tokens;
  }

  // Prefill-side metrics shipped back to Decode so its access log can attribute the
  // disaggregation cost end-to-end (the Prefill node itself prints no access log). This is the
  // single completion chokepoint — OnRequestFinished and OnBarrierDone both funnel here once
  // per request. Durations are Prefill-clock deltas (no cross-node comparison): the three
  // consecutive segments of Prefill residency, arrive → scheduled → finish → send.
  PdPrefillMetrics prefill_metrics;
  if (request != nullptr) {
    prefill_metrics.prefix_cache_hit_tokens = request->prefix_cache_hit_tokens;
    const uint64_t arrive = request->timestamp_in_us;
    const uint64_t scheduled = request->scheduled_time_us.load();
    const uint64_t finish = request->finish_time_us.load();
    const uint64_t now = ProfileTimer::GetCurrentTimeInUs();
    auto delta = [](uint64_t end, uint64_t start) -> uint64_t {
      return (end && start && end >= start) ? end - start : 0;
    };
    prefill_metrics.queue_us = delta(scheduled, arrive);    // 等待: 排队
    prefill_metrics.compute_us = delta(finish, scheduled);  // 计算: prefill forward
    prefill_metrics.send_us = delta(now, finish);           // 发送/收尾: barrier ack + sampler drain
  }

  PeerHealth health;
  if (snapshot_fn) {
    health = snapshot_fn();
  } else {
    health.healthy = (status == PdCompleteStatus::kOk);
  }
  if (connector != nullptr) {
    Status s = connector->SendPrefillComplete(peer_addr, req_id, status, tokens, health, prefill_metrics);
    if (!s.OK()) {
      KLLM_LOG_WARNING << "PdV2PrefillCompletion SendPrefillComplete failed req_id=" << req_id
                       << " err=" << s.GetMessage();
    } else {
      KLLM_LOG_DEBUG << "PdV2PrefillCompletion completed req_id=" << req_id << " status=" << static_cast<int>(status)
                     << " tokens=" << tokens.size();
    }
  }
}

void PdV2PrefillCompletion::OnRequestFinished(std::weak_ptr<SharedState> state_weak,
                                              std::shared_ptr<PdV2ConnectorInterface> connector,
                                              std::function<PeerHealth()> snapshot_fn, uint64_t req_id) {
  auto state = state_weak.lock();
  if (!state) {
    return;
  }
  std::shared_ptr<Request> request;
  std::string peer_addr;
  PdCompleteStatus final_status = PdCompleteStatus::kOk;
  std::vector<int> final_tokens;
  bool ready = false;
  {
    std::lock_guard<std::mutex> lk(state->mu);
    auto it = state->alive.find(req_id);
    if (it == state->alive.end()) {
      KLLM_LOG_WARNING << "PdV2PrefillCompletion::OnRequestFinished: req_id " << req_id
                       << " not found in alive table; dropped";
      return;
    }
    auto& entry = it->second;
    if (entry.request) {
      if (!entry.request->finish_status.OK()) {
        entry.token_status = PdCompleteStatus::kFailed;
        entry.tokens_ready = true;
        entry.barrier_done = true;
        ready = TryCompleteUnlocked(state.get(), req_id, &request, &peer_addr, &final_status, &final_tokens);
      } else {
        std::lock_guard<std::mutex> ol(entry.request->output_mutex);
        const auto& full_output = entry.request->output_tokens;
        const size_t input_size = entry.request->input_tokens.size();
        if (full_output.size() > input_size) {
          entry.tokens.assign(full_output.begin() + input_size, full_output.end());
        } else {
          entry.tokens.clear();
        }
        entry.token_status = entry.tokens.empty() ? PdCompleteStatus::kFailed : PdCompleteStatus::kOk;
        entry.tokens_ready = true;
        ready = TryCompleteUnlocked(state.get(), req_id, &request, &peer_addr, &final_status, &final_tokens);
      }
    }
  }
  if (ready) {
    DoSendPrefillComplete(connector, snapshot_fn, request, peer_addr, final_status, std::move(final_tokens), req_id);
  }
}

size_t PdV2PrefillCompletion::NumAdmittedForTest() const {
  std::lock_guard<std::mutex> lk(state_->mu);
  return state_->num_admitted;
}

size_t PdV2PrefillCompletion::NumCompletedForTest() const {
  std::lock_guard<std::mutex> lk(state_->mu);
  return state_->num_completed;
}

size_t PdV2PrefillCompletion::NumAliveForTest() const {
  std::lock_guard<std::mutex> lk(state_->mu);
  return state_->alive.size();
}

}  // namespace pd_v2
}  // namespace ksana_llm
