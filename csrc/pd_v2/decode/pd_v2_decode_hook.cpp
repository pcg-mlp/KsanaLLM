/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/decode/pd_v2_decode_hook.h"

#include <algorithm>

#include "pd_v2/decode/local_load.h"
#include "profiler/timer.h"
#include "runtime/infer_request.h"
#include "utils/logger.h"

namespace ksana_llm {
namespace pd_v2 {

PdV2DecodeHook::PdV2DecodeHook(std::shared_ptr<PdV2ConnectorInterface> connector, int hook_idx,
                               int local_complete_threshold, SelectorPolicy selector_policy,
                               std::function<ScheduleLoadStat()> load_stat_fn)
    : connector_(std::move(connector)),
      hook_idx_(hook_idx),
      local_complete_threshold_(local_complete_threshold),
      selector_policy_(selector_policy),
      load_stat_fn_(std::move(load_stat_fn)) {
  KLLM_CHECK(connector_ != nullptr);
}

void PdV2DecodeHook::SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) {
  async_callback_ = std::move(callback);
}

size_t PdV2DecodeHook::PendingCountForTest() const {
  std::lock_guard<std::mutex> lk(mu_);
  return completion_results_.size();
}

SamplingConfigWire PdV2DecodeHook::MakeWireSamplingConfig(const SamplingConfig& sc) {
  SamplingConfigWire wire;
  wire.topk = sc.topk;
  wire.topp = sc.topp;
  wire.temperature = sc.temperature;
  wire.repetition_penalty = sc.repetition_penalty;
  // Prefill must produce exactly the tokens Decode needs to bootstrap:
  // 1 main token + max(0, ptp_token_num) MTP/PTP drafts. ptp_token_num
  // == -1 means "use global default" in ksana_llm semantics; for the
  // wire we collapse that to 0 drafts so Prefill doesn't have to know
  // the Decode-side global.
  const int drafts = std::max(0, sc.ptp_token_num);
  wire.max_new_tokens = 1 + drafts;
  wire.stop_token_ids.assign(sc.stop_token_ids.begin(), sc.stop_token_ids.end());
  wire.stop_strings = sc.stop_strings;
  wire.ignore_eos = sc.ignore_eos;
  wire.json_schema = sc.json_schema;
  wire.regex = sc.regex;
  wire.grammar = sc.grammar;
  wire.structural_tag = sc.structural_tag;
  wire.enable_structured_output = sc.enable_structured_output;
  wire.enable_thinking = sc.enable_thinking;
  wire.ptp_token_num = sc.ptp_token_num;
  return wire;
}

ReqIdToNextStep PdV2DecodeHook::Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) {
  ReqIdToNextStep result;
  for (const auto& req : requests) {
    // pd_v2 warmup bypass: Decode warmup requests carry req_id == -1
    // (set by inference_engine.cpp::DoWarmupRun for pd_v2 DECODE role,
    // mirroring V1 PREFILL convention). Skip the SubmitPrefillRequest
    // round-trip and run warmup locally so Decode bring-up doesn't
    // depend on a peer Prefill being alive at warmup time, and so the
    // partial PD state propagation issues (kv_from_remote / kv_cached
    // not flowing to executor) don't crash warmup before any user
    // request arrives.
    if (req->req_id < 0) {
      KLLM_LOG_INFO << "PdV2DecodeHook: req_id=" << req->req_id << " (warmup) → bypass Prefill, run locally";
      continue;
    }

    // Defensive: if the prefill sequence is empty, the request has no prompt
    // data to forward to Prefill. Let it fall through to the normal local path
    // rather than sending an empty stage payload that causes Prefill to
    // crash on empty forwarding_tokens (model_runner GetHiddenShape).
    // We key on prefilling_tokens_ (== input_tokens for fresh requests, ==
    // accumulated [prompt + generated] for a pd_v2 recompute restart) so the
    // restart path re-prefills the whole accumulated sequence and preserves
    // already-generated tokens.
    if (req->GetPrefillingTokens().empty()) {
      KLLM_LOG_WARNING << "PdV2DecodeHook: req_id=" << req->req_id
                       << " has empty prefilling tokens, bypassing PD dispatch";
      continue;
    }

    // global cache 全命中 (V1 standalone 行为下): GlobalCacheFetcher 已经
    //   - 把 KV 数据写入 kv_cache_blocks (executor QueryAndFetch 阶段)
    //   - 把 infer_stage flip 为 kDecode
    //   - 已 AppendCachedTokens (有 cached tokens 时) 或 HandleFullPrefixCacheHit
    // 此时 peer Prefill 没活可干, 调 SubmitPrefillRequest 反而会让 peer 重算并 RDMA
    // 覆盖已经填好的 KV (kv_from_remote 也会撞 AppendCachedTokens 的 first token).
    // 直接 bypass, 让 EDS 按 infer_stage=kDecode 路由到 decoding_reqs.
    if (req->infer_stage == InferStage::kDecode) {
      KLLM_LOG_DEBUG << "PdV2DecodeHook: req_id=" << req->req_id
                     << " infer_stage=kDecode (global cache full hit), bypass peer Prefill";
      continue;
    }

    // pd_v2 short-request fast path: when the prefix-cache hit on this
    // Decode node is large enough that the remaining prefill workload
    // is small (≤ local_complete_threshold tokens), forwarding to peer
    // Prefill is net negative — control-plane round-trip + KV transfer
    // overhead exceeds the local prefill cost. Skip Submit entirely;
    // return kContinue with no action. The request flows through normal
    // ksana_llm prefill+decode locally on this node, with kFlash on the
    // remaining (input − prefix_cache) tokens, then steady-state decode.
    //
    // Critical: kv_from_remote stays false on this path. The
    // ForwardRequest::GetType() fast-path won't fire, so GetType falls
    // through to the standard `GetInputIdsLength <= threshold &&
    // kv_cached_token_num > 0` check — which routes correctly for both
    // small-residual prefill (kFlash on the small tail) and subsequent
    // decode steps (kPage). LaunchPlanningTask:314 SetKvCachedTokenNum
    // also runs normally (the !kv_from_remote branch).
    //
    // 0 disables (always forward to Prefill). Default deployment opts
    // in via EngineConfig.local_complete_threshold.
    //
    // global cache 部分命中也应算入 prefix: fetcher 可能命中本地 prefix 之外的 token, 此时
    // remaining 真实长度 = input - max(local_prefix, fetcher_prefix). 取最大值即可统一覆盖
    // (a) 没启 global cache (fetcher_prefix=0, 退回 prefix_cache_len),
    // (b) global cache miss (fetcher_prefix=0),
    // (c) 部分命中 (fetcher_prefix > prefix_cache_len),
    // (d) 全命中 (上面 infer_stage=kDecode bypass 已 short-circuit, 不会到这里).
    const int prefix_len =
        std::max(std::max(0, req->prefix_cache_len), static_cast<int>(req->GetPlanningWorkload().prefill_start_offset));
    // Use the prefill sequence (accumulated on a recompute restart) so the
    // short-request fast path measures the real remaining prefill workload.
    const std::vector<int>& prefill_tokens = req->GetPrefillingTokens();
    const int remaining = static_cast<int>(prefill_tokens.size()) - prefix_len;
    const bool use_linucb = (selector_policy_ == SelectorPolicy::kLinUCB);
    // remaining <= 0: Decode-side prefix cache covers the entire prompt —
    // Prefill has zero work to do and zero KV to RDMA back, so the round-
    // trip is pure waste. Bypass unconditionally (independent of policy).
    if (remaining <= 0) {
      KLLM_LOG_DEBUG << "PdV2DecodeHook: req_id=" << req->req_id << " prefill_tokens=" << prefill_tokens.size()
                     << " prefix_cache=" << prefix_len << " remaining=" << remaining
                     << " → bypass Prefill (no residual work), run locally";
      continue;
    }
    // Static threshold soft-bypass: small residual is cheaper as local prefill
    // than the Prefill round-trip + KV transfer. Only when NOT using the
    // bandit — under kLinUCB the "small remaining" case is folded into the
    // router's remaining_small feature, so the bandit owns the decision.
    if (!use_linucb && local_complete_threshold_ > 0 && remaining <= local_complete_threshold_) {
      KLLM_LOG_DEBUG << "PdV2DecodeHook: req_id=" << req->req_id << " prefill_tokens=" << prefill_tokens.size()
                     << " prefix_cache=" << prefix_len << " remaining=" << remaining
                     << " threshold=" << local_complete_threshold_ << " → bypass Prefill, run locally";
      // No NextStep entry → schedule processor falls through to
      // default kContinue path → req goes to prefilling_reqs and runs
      // standard local prefill + decode. (Hooks return entries only
      // for reqs they intercept; absent entry == "no opinion".)
      continue;
    }

    // LOCAL-vs-REMOTE routing. The connector decides: non-kLinUCB policies
    // make RouteRequest equivalent to CanAccept (forward when a peer exists,
    // else fail); kLinUCB consults the contextual bandit. Build the bandit's
    // local-load context only when kLinUCB is active.
    LocalArmContext arm_ctx;
    arm_ctx.remaining_prefill_tokens = static_cast<uint32_t>(remaining);
    if (use_linucb) {
      const ScheduleLoadStat stat = load_stat_fn_ ? load_stat_fn_() : ScheduleLoadStat{};
      const LocalLoadView ld = ReadLocalLoad(context, static_cast<size_t>(req->attn_dp_group_id), stat.step_token_num,
                                             stat.step_latency_us, stat.step_latency_us_inst);
      arm_ctx.running_reqs = ld.running_reqs;
      arm_ctx.step_latency_us = ld.step_latency_us;
      arm_ctx.step_latency_us_inst = ld.step_latency_us_inst;
    }
    const std::optional<RouteArm> arm = connector_->RouteRequest(static_cast<uint64_t>(req->req_id), arm_ctx);
    if (!arm.has_value()) {
      KLLM_LOG_WARNING << "PdV2DecodeHook: RouteRequest declined req_id=" << req->req_id << ", no healthy peer";
      result[req->req_id] = NextStep{NextStepAction::kFailAndEnd, 0};
      continue;
    }
    if (*arm == RouteArm::kLocal) {
      KLLM_LOG_DEBUG << "PdV2DecodeHook: req_id=" << req->req_id << " remaining=" << remaining
                     << " → LinUCB kLocal, run remaining prefill locally";
      // Connector recorded the deferred LOCAL reward; no in_flight_ entry, no
      // SubmitPrefillRequest. Fall through to normal local prefill + decode.
      continue;
    }
    // kRemotePeer: connector admitted the request (in_flight_ recorded, peer
    // chosen by PeerSelector). Proceed to build + send the PrefillRequest.

    // 仅 kRemotePeer 路径会写入 connector 状态，析构时需要清理 in_flight_ 并释放 stage credits。
    // kLocal 的延迟 reward 由 SampleLocalReward 消费，bypass 路径也没有 connector 状态需要释放。
    if (connector_ != nullptr) {
      auto* connector_raw = connector_.get();
      const uint64_t req_id_capture = static_cast<uint64_t>(req->req_id);
      req->AddOnDestroy([connector_raw, req_id_capture]() { connector_raw->Release(req_id_capture); });
    }

    SamplingConfigWire wire = MakeWireSamplingConfig(req->sampling_config);

    // pd_v2 prefix-cache hint to peer Prefill. The cached prefix length combines:
    //   - 本地 prefix cache 命中 (req->prefix_cache_len)
    //   - global cache 部分命中 (planning_workload.prefill_start_offset 由 fetcher
    //     UpdateRequestWorkload 写入, 已包含本地 + global 的连续 prefix 长度)
    // 取两者最大值: planning_workload.prefill_start_offset 在 fetcher 跑完后是权威值;
    // 当 fetcher 没跑或返回 0 命中时, 退回到 prefix_cache_len. peer Prefill 用这个 offset
    // 跳过已经被 fetcher / 本地 prefix 写好的 block, 避免重算 + RDMA 覆盖.
    const int local_prefix = std::max(0, req->prefix_cache_len);
    const int fetcher_prefix = static_cast<int>(req->GetPlanningWorkload().prefill_start_offset);
    const uint32_t decode_prefix_cache_offset = static_cast<uint32_t>(std::max(local_prefix, fetcher_prefix));
    const uint32_t block_token_num = req->block_token_num > 0 ? static_cast<uint32_t>(req->block_token_num) : 1u;
    const uint32_t cached_block_count = decode_prefix_cache_offset / block_token_num;

    // GlobalCacheQueryHook 是 KV block 分配的单一所有者 (hook_idx=0)。pd_v2 Decode 角色下
    // 它在 alloc 失败时会自行挂起重试 (defer_admission_on_block_full), 因此能走到这里的 req
    // 一定已经分好了 block。kv_cache_blocks 为空属于不变量被破坏 — fail-fast 而不是带着空
    // RDMA target 列表去 SubmitPrefillRequest (会让 peer Prefill 在空 forwarding_tokens 上崩溃)。
    if (req->kv_cache_blocks.empty()) {
      KLLM_LOG_ERROR << "PdV2DecodeHook: req_id=" << req->req_id
                     << " reached PD dispatch with empty kv_cache_blocks (GlobalCacheQueryHook should own alloc); "
                        "failing request";
      result[req->req_id] = NextStep{NextStepAction::kFailAndEnd, 0};
      continue;
    }

    Status s = connector_->SubmitPrefillRequest(static_cast<uint64_t>(req->req_id), prefill_tokens,
                                                req->kv_cache_blocks, wire, req->attn_dp_group_id, cached_block_count,
                                                decode_prefix_cache_offset, &PdV2DecodeHook::OnPrefillComplete,
                                                /*ctx=*/this);

    if (!s.OK()) {
      KLLM_LOG_ERROR << "PdV2DecodeHook: SubmitPrefillRequest failed for req_id=" << req->req_id << ": "
                     << s.GetMessage();
      // Roll back admission so the connector doesn't leak in_flight state.
      connector_->Release(static_cast<uint64_t>(req->req_id)).GetMessage();
      result[req->req_id] = NextStep{NextStepAction::kFailAndEnd, 0};
      continue;
    }

    // pd_v2 RDMA flight window opens here. kv_cache_blocks is now wired
    // into peer Prefill's RDMA target list; recompute on this req between
    // now and the matching OnPrefillComplete callback would race with
    // in-flight RDMA writes and corrupt the victim's KV bytes.
    // The state-C req should sit in preprocessing_reqs (kWaitAsyncResult
    // routes it there) and be invisible to the prefilling_reqs / decoding
    // _reqs victim scans in continuous_batching_backend. The flag is a
    // bug-detector for that invariant: if backend ever sees this true,
    // it logs an ERROR (state machine drifted).
    req->pd_v2.awaiting_prefill = true;
    // T2: request dispatched to peer Prefill (RDMA flight window opens here).
    StampOnce(req->pd_v2.prefill_dispatched_time_us);
    // PD 链路关键点[1/4]: decode 已受理请求并派发给 peer Prefill (保留 INFO 便于阻塞定位)。
    KLLM_LOG_INFO << "PdV2DecodeHook: req_id=" << req->req_id << " submitted to peer Prefill, waiting for complete";
    result[req->req_id] = NextStep{NextStepAction::kWaitAsyncResult, 0};
  }
  return result;
}

void PdV2DecodeHook::OnPrefillComplete(uint64_t req_id, PdCompleteStatus status,
                                       const std::vector<int>& generated_tokens,
                                       const PdPrefillMetrics& prefill_metrics, const std::string& prefill_peer,
                                       void* ctx) {
  auto* self = static_cast<PdV2DecodeHook*>(ctx);
  if (self == nullptr) {
    KLLM_LOG_ERROR << "PdV2DecodeHook::OnPrefillComplete: null ctx for req_id=" << req_id;
    return;
  }
  {
    std::lock_guard<std::mutex> lk(self->mu_);
    PendingResult pr;
    pr.status = status;
    pr.generated_tokens = generated_tokens;
    pr.prefill_metrics = prefill_metrics;
    pr.prefill_peer = prefill_peer;
    // T3: decode-clock arrival of PrefillComplete (KV/first token). Stamped here (on the
    // connector dispatch thread, the true network-landing moment), not in UpdateAsyncResult
    // which runs a scheduler tick later.
    pr.complete_time_us = ProfileTimer::GetCurrentTimeInUs();
    self->completion_results_[req_id] = std::move(pr);
  }

  // Wake the EDS main loop so it dispatches UpdateAsyncResult on its
  // own thread. We pass an AsyncTaskResult skeleton; the hook drains
  // its own side map by req_id, so task_id is unused here.
  if (self->async_callback_) {
    EventAsyncTaskFinishedInfo info;
    AsyncTaskResult tr;
    tr.hook_idx = self->hook_idx_;
    tr.task_id = 0;
    tr.task_ret_state = (status == PdCompleteStatus::kOk) ? 0 : 1;
    info.task_results.push_back(tr);
    self->async_callback_(info);
  }
}

ReqIdToNextStep PdV2DecodeHook::UpdateAsyncResult(const std::vector<AsyncTaskResult>& /*finished_tasks*/,
                                                  ScheduleContext& context) {
  ReqIdToNextStep result;

  // Block allocation + admission backpressure is owned entirely by
  // GlobalCacheQueryHook (hook_idx=0). This hook only reacts to peer-Prefill
  // completions (OnPrefillComplete → completion_results_); it neither allocates
  // blocks nor defers admission.

  // Snapshot + clear the side map under the lock. We then mutate req
  // fields outside the lock; reqs are looked up via ScheduleContext.
  std::unordered_map<uint64_t, PendingResult> drained;
  {
    std::lock_guard<std::mutex> lk(mu_);
    drained.swap(completion_results_);
  }

  for (auto& kv : drained) {
    const uint64_t req_id = kv.first;
    const PendingResult& pr = kv.second;

    auto req = context.GetReqById(static_cast<int64_t>(req_id));
    if (req == nullptr) {
      KLLM_LOG_WARNING << "PdV2DecodeHook::UpdateAsyncResult: req_id=" << req_id
                       << " no longer in schedule state, dropping";
      continue;
    }

    KLLM_LOG_DEBUG << "PdV2DecodeHook::UpdateAsyncResult: req_id=" << req_id
                   << " status=" << static_cast<int>(pr.status) << " tokens=" << pr.generated_tokens.size()
                   << " pd_v2_awaiting_prefill=" << req->pd_v2.awaiting_prefill
                   << " pd_v2_pending_stop=" << req->pd_v2.pending_stop
                   << " infer_stage=" << static_cast<int>(req->infer_stage);

    // Drop late PrefillComplete if the req was already stopped/finished.
    if (req->IsStopped() || req->finished) {
      req->pd_v2.awaiting_prefill = false;
      req->pd_v2.pending_stop = false;
      KLLM_LOG_WARNING << "PdV2DecodeHook::UpdateAsyncResult: req_id=" << req_id
                       << " already terminated (is_stopped=" << req->IsStopped() << " finished=" << req->finished
                       << "); dropping late PrefillComplete without writing req fields";
      result[req->req_id] = NextStep{NextStepAction::kFailAndEnd, 0};
      continue;
    }

    req->pd_v2.awaiting_prefill = false;

    // T3: record the decode-clock arrival of PrefillComplete (stamped on the dispatch thread
    // in OnPrefillComplete), and stash the peer Prefill's own-clock metrics for the access log.
    req->pd_v2.prefill_complete_time_us.store(pr.complete_time_us);
    req->pd_v2.pf_prefix_cache_hit_tokens.store(pr.prefill_metrics.prefix_cache_hit_tokens);
    req->pd_v2.pf_queue_us.store(pr.prefill_metrics.queue_us);
    req->pd_v2.pf_compute_us.store(pr.prefill_metrics.compute_us);
    req->pd_v2.pf_send_us.store(pr.prefill_metrics.send_us);
    // Selected peer Prefill engine, via the req_ctx side-channel (same path as mtp stats):
    // KsanaPythonOutput reads it back for the access log.
    if (req->req_ctx && !pr.prefill_peer.empty()) {
      (*req->req_ctx)["pd_v2_prefill_peer"] = pr.prefill_peer;
    }

    // User cancel / EDS timeout while WAITING_KVCACHE, or non-OK Prefill status.
    const bool pending_stop = req->pd_v2.pending_stop;
    if (pending_stop || pr.status != PdCompleteStatus::kOk || pr.generated_tokens.empty()) {
      req->pd_v2.pending_stop = false;
      KLLM_LOG_ERROR << "PdV2DecodeHook: req_id=" << req_id
                     << " prefill terminal, status=" << static_cast<int>(pr.status) << " pending_stop=" << pending_stop;
      result[req->req_id] = NextStep{NextStepAction::kFailAndEnd, 0};
      continue;
    }

    // Splice generated tokens into req. Order matters: append output
    // first so streaming sees the new token before we flip infer_stage
    // (which makes EDS route to decoding_reqs on the next sweep).
    {
      std::lock_guard<std::mutex> ol(req->output_mutex);
      req->output_tokens.push_back(pr.generated_tokens[0]);
    }
    if (pr.generated_tokens.size() > 1) {
      // MTP drafts produced by Prefill's first-step + MTP forward.
      req->draft_tokens.mtp.assign(pr.generated_tokens.begin() + 1, pr.generated_tokens.end());
    }
    // pd_v2 cache contract: peer Prefill RDMA-wrote K/V for the prompt
    // (positions 0..N-1). The first token Prefill sampled (= Paris) is
    // returned via the wire and goes through `generated_tokens` so that
    // InferRequest::LaunchPlanningTask appends it to
    // forwarding_tokens for the FIRST decode-step query.
    //
    // forwarding_tokens MUST equal input_tokens here, with kv_cached =
    // input_tokens.size() — that pair tells the rest of the pipeline:
    // "the prompt is fully cached; the next forward query is whatever
    // gets appended". After this hook returns, the EDS schedule
    // processor moves the req to decoding_reqs (the schedule processor
    // already gates on infer_stage == kDecode), then
    // LaunchPlanningTask runs:
    //   - `SetKvCachedTokenNum(forwarding_tokens.size())` is
    //     gated on !kv_from_remote (won't clobber our 6)
    //   - `forwarding_tokens.insert(generated_tokens)` appends
    //     [Paris] → forwarding_tokens becomes [prompt, Paris] (size 7)
    //   - GetInputIdsLength = 7 - 6 = 1 → kPage path
    // ForwardRequest::GetType() also checks kv_from_remote first as a
    // safety net.
    // Use prefilling_tokens_ (== input_tokens for a fresh request, ==
    // accumulated [prompt + generated] for a pd_v2 recompute restart). On
    // restart the peer re-prefilled the whole accumulated sequence and the
    // returned token is the genuine next token, so output_tokens.push_back
    // above appends it without duplicating already-generated tokens.
    const std::vector<int>& prefill_tokens = req->GetPrefillingTokens();
    req->forwarding_tokens = prefill_tokens;
    req->generated_tokens.assign(pr.generated_tokens.begin(), pr.generated_tokens.end());
    req->pd_v2.kv_from_remote = true;
    req->SetKvCachedTokenNum(prefill_tokens.size());
    req->infer_stage = InferStage::kDecode;

    // pd_v2 schedule plumbing fix (2026-05-16):
    // GlobalCacheQueryHook ran on this req at admission and set
    // remaining_workload_ as a PREFILL workload (prefill_token_num =
    // input_tokens.size()). After Prefill RDMA-wrote the KV and we
    // flipped infer_stage = kDecode, that prefill workload is stale —
    // but the schedule processor still copies remaining_workload_ →
    // planning_workload_ → planning_task_.workload at scheduling time.
    // LaunchPlanningTask then sees prefill_token_num > 0, enters its
    // "prefix cache hit" branch, calls SetKvCachedTokenNum(prefill_start_offset=0)
    // which CLOBBERS the kv_cached_token_num=N we just set.
    // Net effect: executor sees kv_cached=0 / forwarding=N / kv_from_remote=1,
    // paged_mla runs over N q_tokens instead of 1, output is 1 token
    // then nothing.
    //
    // Fix: zero out remaining prefill work AND set prefill_start_offset =
    // input_tokens.size() to keep SetPlanningTask's invariant
    //   remaining.prefill_token_num + remaining.prefill_start_offset
    //     == prefilling_tokens_.size()
    // happy (line 251 of infer_request.cpp). Plus seed generated_token_num
    // so the decode path at LaunchPlanningTask:356 has resource_ready_token_num
    // covering the spliced first token.
    ScheduleTaskWorkload remaining = req->GetRemainingWorkload();
    remaining.prefill_token_num = 0;
    remaining.prefill_start_offset = prefill_tokens.size();
    remaining.generated_token_num = req->generated_tokens.size();
    remaining.draft_token_num = 0;
    req->SetRemainingWorkload(remaining);

    // TTFT critical: wake the streaming iterator immediately, do NOT
    // wait for the next decode forward. NotifyStep is null-safe
    // (`if (step_waiter)`), so non-streaming requests no-op cleanly.
    req->NotifyStep();

    KLLM_LOG_SCHEDULER << "PdV2DecodeHook: req_id=" << req_id << " completed with " << pr.generated_tokens.size()
                       << " tokens, flipped to kDecode";
    result[req->req_id] = NextStep{NextStepAction::kContinue, 0};
  }
  return result;
}

}  // namespace pd_v2
}  // namespace ksana_llm
