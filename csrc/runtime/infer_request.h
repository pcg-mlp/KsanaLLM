/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "cache_manager/cache_manager_interface.h"
#include "pd_v2/public/pd_v2_types.h"
#include "profiler/timer.h"
#include "runtime/infer_stage.h"
#include "runtime/model_instance.h"
#include "runtime/speculative/common/draft_tokens.h"
#include "runtime/structured_generation/structured_generator_interface.h"
#include "utils/calc_intvec_hash.h"
#include "utils/request.h"
#include "utils/status.h"
#include "utils/tensor.h"
#include "utils/waiter.h"

namespace ksana_llm {

// Atomically stamp current monotonic time (us) into `slot` once.
// Subsequent calls are no-ops, so preempt/recompute paths don't overwrite the first stamp.
// Defined out-of-line in infer_request.cpp for accurate coverage attribution.
void StampOnce(std::atomic<uint64_t> &slot);

// Workload of a task
struct ScheduleTaskWorkload {
  size_t prefill_token_num;
  size_t prefill_start_offset;
  size_t generated_token_num;
  size_t draft_token_num;
  size_t sampling_token_num;
  ScheduleTaskWorkload() { Reset(); }
  bool IsEmpty() const { return prefill_token_num == 0 && generated_token_num == 0 && draft_token_num == 0; }
  size_t GetTokenNum() const { return prefill_token_num + generated_token_num + draft_token_num; }
  void Reset() {
    prefill_token_num = 0;
    prefill_start_offset = 0;
    generated_token_num = 0;
    draft_token_num = 0;
    sampling_token_num = 0;
  }

  const std::string ToString() const {
    if (IsEmpty()) {
      return "()";
    }

    std::ostringstream ss;
    ss << "( ";
    if (prefill_token_num > 0) {
      ss << "prefill_token_num=" << prefill_token_num << ", prefill_start_offset=" << prefill_start_offset;
    }
    if (generated_token_num > 0) {
      ss << " generated_token_num=" << generated_token_num;
    }
    if (draft_token_num > 0) {
      ss << " draft_token_num=" << draft_token_num;
    }
    if (sampling_token_num > 0) {
      ss << " sampling_token_num=" << sampling_token_num;
    }
    ss << " )";
    return ss.str();
  }
};

struct ScheduleTask {
  ScheduleTaskWorkload workload;
  bool IsEmpty() const { return workload.IsEmpty(); }
  void Reset() { workload.Reset(); }
};

// pd_v2 (Prefill/Decode disaggregation v2) per-request state, extracted out of
// InferRequest to keep the core request free of disaggregation-specific
// clutter. Owned value fields carry the Decode-side KV routing and the
// Decode state-machine handshake flags; the trailing atomic references alias
// the backing Request's phase-timestamp / peer-Prefill-metric counters
// (stamped via StampOnce on the Decode node, read at access-log time).
//
// Non-pd_v2 requests leave every owned field at its default and never touch the
// referenced atomics, so V1 and Prefill-side paths are unaffected.
struct PdV2RequestContext {
  // Binds the atomic references to `request`'s backing fields. `request` must
  // outlive this context (guaranteed: InferRequest holds it for its lifetime).
  explicit PdV2RequestContext(Request &request);

  // 含引用成员，别名 backing Request 的字段。禁止拷贝/移动，避免 memcpy/std::move
  // 后引用悬空（这类错误在运行时极难排查）。
  PdV2RequestContext(const PdV2RequestContext &) = delete;
  PdV2RequestContext &operator=(const PdV2RequestContext &) = delete;
  PdV2RequestContext(PdV2RequestContext &&) = delete;
  PdV2RequestContext &operator=(PdV2RequestContext &&) = delete;

  // Decode-side KV pool routing for cross-node Prefill→Decode KV RDMA write.
  // Populated on the Prefill engine by PdV2PrefillIntakeWrapper when it
  // deserializes an IncomingPrefillRequest, then carried down to the Prefill
  // executor via ScheduleOutput. Empty vector means "not a pd_v2 request,
  // executor's per-layer KV-write hook is a no-op".
  std::vector<pd_v2::DecodeTargetLite> decode_targets;

  // Mooncake server_name of the originating Decode engine. Used by the Prefill
  // executor as the openSegment() target in submitTransfer. Empty when
  // decode_targets is empty.
  std::string decode_engine_inference_addr;

  // Decode-side attn_dp_group_id this request belongs to. The Prefill executor
  // needs it to compose the per-rank server_name suffix `.exec.r<device_id>`
  // where device_id = decode_attn_dp_group_id * attn_TP_decode +
  // decode_attn_tp_rank_in_group. Default 0 is the historical behavior (single
  // attn_dp group), so non-pd_v2 paths and Decode attn_DP=1 are unaffected.
  uint32_t decode_attn_dp_group_id = 0;

  // When true, this request's K/V for the prompt has been written by a peer
  // Prefill node (Mooncake RDMA into the local KV pool) and the hook has
  // already pre-populated forwarding_tokens / kv_cached_token_num /
  // generated_tokens / infer_stage to match. The scheduling pipeline (most
  // notably LaunchPlanningTask, ResetPrefillingTokens, ForwardRequest::GetType)
  // MUST trust the hook-set state and not re-derive cache state from
  // forwarding_tokens.size() or assume "first forward = prefill". If we let the
  // default machinery run on these reqs, Decode would re-run prefill on the
  // prompt and overwrite the bytes the peer Prefill RDMA'd over, producing
  // degenerate output. Stays true for the lifetime of the req — once true, it
  // never flips back.
  //
  // Reuses the name from a long-deleted V1 field of similar intent (memory: R1
  // 撤回 HBP / kv_from_remote). Semantics are NEW under pd_v2 and unrelated to
  // the V1 hidden-buffer flow.
  bool kv_from_remote = false;

  // Decode-side: true while a request has been Submitted to peer Prefill but
  // the PrefillComplete callback has not fired yet. During this RDMA flight
  // window, kv_cache_blocks is already wired into peer Prefill's RDMA target
  // list — recompute / swap-out on this req would race with in-flight RDMA
  // writes and corrupt the victim's KV bytes. Set in PdV2DecodeHook::Process
  // right before returning kWaitAsyncResult, cleared in UpdateAsyncResult right
  // where kv_from_remote is flipped true (or in the connector::Release / fail
  // paths). Always false on V1 and Prefill-side paths (no harm).
  bool awaiting_prefill = false;

  // Decode-side: user cancel / EDS timeout arrived while awaiting_prefill
  // (WAITING_KVCACHE). Do not release KV blocks until PrefillComplete — RDMA
  // may still be writing. The decode hook applies Stop on the next
  // UpdateAsyncResult after OnPrefillComplete.
  bool pending_stop = false;

  // Decode-side: set when a kv_from_remote request is preempted via recompute
  // and re-injected through EventDrivenScheduleProcessor::AddInferRequest for a
  // full PD restart. swap is being removed, so the only viable preempt path for
  // Decode reqs is to release the local KV and re-fetch from a peer Prefill. To
  // preserve already-generated tokens we re-prefill the accumulated sequence
  // (output_tokens), so OnEventAddRequest MUST NOT reset
  // output_tokens = input_tokens for these reqs. The flag is consumed (cleared)
  // by OnEventAddRequest after ResetPrefillingTokens makes
  // prefilling_tokens_ = output_tokens. Always false on V1 / fresh requests.
  bool restart_resubmit = false;

  // Decode-side phase timestamps, references to Request fields. Stamped via StampOnce().
  std::atomic<uint64_t> &blocks_allocated_time_us;
  std::atomic<uint64_t> &prefill_dispatched_time_us;
  std::atomic<uint64_t> &prefill_complete_time_us;

  // Decode-side Prefill metrics received over the wire, references to Request fields.
  std::atomic<int> &pf_prefix_cache_hit_tokens;
  std::atomic<uint64_t> &pf_queue_us;
  std::atomic<uint64_t> &pf_compute_us;
  std::atomic<uint64_t> &pf_send_us;
};

// The infer request, it is the unit of batch manager's scheduler.
class InferRequest {
 public:
  InferRequest(std::shared_ptr<Request> &request, const int index);
  ~InferRequest();

  // 注册 InferRequest 析构时触发的回调。pd_v2 用它在调度器释放最后一个 shared_ptr 时清理
  // connector 状态；回调应保持轻量且不抛异常，析构函数会兜底吞掉异常。
  void AddOnDestroy(std::function<void()> cb) {
    if (cb) on_destroy_callbacks_.push_back(std::move(cb));
  }

  void SetReqGroup(const std::vector<std::shared_ptr<InferRequest>> &beam_search_infer_group) {
    req_group = beam_search_infer_group;
  }

  // Clear the group of requests.
  void ClearReqGroup() { req_group.clear(); }

  // Notify after request finished.
  void Notify();

  // Notify after step finished.
  void NotifyStep();

  // Get this infer request's KV occupied devices.
  std::vector<int> GetKVOccupiedDevices();

  std::string PrintKVBlockIds(bool print_details = false) const;

  std::string ToString(bool print_details = false) const;

  friend std::ostream &operator<<(std::ostream &os, const InferRequest &req);

 public:
  // In a step, forward() takes sequence and query as input.
  // sequence is tokens can be input tokens + generated tokens + draft tokens).
  // query is tokens processed in this step. It is last part of sequence.
  //      It can be part of input tokens, or generated tokens + draft tokens.

  // Get sequence for inflight step, all tokens have kv cache space
  const std::vector<int> &GetInflightSequence() const;
  // Get sequence length for inflight step
  size_t GetInflightSequenceLen() const;
  // Get query len in inflight step
  size_t GetInflightQueryLen() const;
  // Get the number of tokens to be sampled in inflight step
  size_t GetInflightSamplingTokenNum() const;
  // After inflight step executed, adjust sequence accoding to generation result
  // size_t SetInFlightGenResult(const GenResult& result);

  // Get sequence length for planning step, all tokens need kv cache
  size_t GetPlanningSequenceLen() const;
  // Get query len in planning step
  size_t GetPlanningQueryLen() const;
  // Get/Set the number of tokens to be sampled in planning step
  size_t GetPlanningSamplingTokenNum() const;

  void SetPlanningGeneratedTokenNum(size_t num);
  void SetPlanningDraftTokenNum(size_t num);

  void SetKvCachedTokenNum(size_t num);

  void SetCacheHitStatus(size_t shared_token_num, bool is_first_prefill_step = true);

 public:
  // The req id of the user's request.
  int64_t req_id;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  size_t sampling_token_num = kStepGenerateTokenNum;

  // Record the number of tokens sampled in the previous step.
  size_t last_step_token_num = sampling_token_num;

  // The input tokens.
  std::vector<int> &input_tokens;

  // Embedding slice used to refit input embedding
  EmbeddingSlice &input_refit_embedding;

  // The offset for multimodal rotary position embedding, computed in prefill phase by Python plugin,
  // and used in decode phase.
  int64_t mrotary_embedding_pos_offset = 0;
  int64_t xdrotary_embedding_pos_offset = 0;

  // output_tokens is used during computation. When split fuse is enabled, output_tokens contains only the split
  // part. This variable is dynamically updated based on the current computation phase and may not always represent the
  // complete output.
  std::vector<int> &output_tokens;

  // draft token generated by MTP and Trie
  DraftTokens draft_tokens;

  // Suggested number of draft tokens to generate, determined by the scheduler
  size_t suggested_draft_num = 0;

  // accepted draft tokens
  std::vector<int> accepted_tokens;

  // draft token num in forwarding_tokens
  size_t forwarding_tokens_draft_num = 0;

  // Cumulative MTP/draft acceptance stats over the request lifetime, for access-log hit-rate.
  size_t mtp_hit_token_num_ = 0;    // accepted draft tokens
  size_t mtp_draft_token_num_ = 0;  // proposed draft tokens

  // token generated by model, complete new tokens in a step are (draft_tokens - reject_token_num) + generated_tokens
  std::vector<int> generated_tokens;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::pair<int, float>>> &logprobs;

  // Store request cache hit status.
  // Prefix cache: (prefix_len, prefix_len), hit [0, prefix_len)
  // Flexible cache v1: (prefix_len, prefix_len + flexible_len), hit [prefix_len, prefix_len + flexible_len)
  // Flexible cache v2: (start1, end1), (start2, end2), ..., hit [start1, end1), [start2, end2), ...
  bool return_cache_stat;
  std::vector<std::pair<size_t, size_t>> &cache_stat;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  std::unordered_map<std::string, TargetDescribe> &request_target;

  // The result of request_target.
  std::unordered_map<std::string, PythonTensor> &response;

  // The sampling config of this request.
  SamplingConfig &sampling_config;

  // The waiter used to notify when request finished.
  std::shared_ptr<Waiter> &waiter;

  // The waiter used to notify when step finished.
  std::shared_ptr<Waiter> &step_waiter;

  // The waiter used to notify when request aborted..
  std::shared_ptr<Waiter> &abort_waiter;

  // Structured generator configuration for structured output
  StructuredGeneratorConfig &structured_generator_config;

  // Whether the request is finished.
  bool &finished;

  // whether the request is aborted.
  std::atomic<bool> &aborted;

  // Whether the req is mock.
  bool is_mock_req = 0;

  // The final status of this request.
  Status &finish_status;

  // Protect parallel access for output token.
  std::mutex &output_mutex;

  std::vector<std::shared_ptr<InferRequest>> req_group;

  // The model instance pointer.
  std::shared_ptr<ModelInstance> model_instance;

  // Different reqs may have different cache managers.
  std::shared_ptr<CacheManagerInterface> cache_manager;

  // data parallel id of this request.
  uint32_t attn_dp_group_id = 0;

  // This is a unique ID for the KV transformer group.
  int64_t kv_comm_request_id = 0;

  // This key for kv transformer group.
  std::string kv_comm_group_key;

  // pd_v2 (Prefill/Decode disaggregation) per-request state. All pd_v2-specific
  // fields (Decode KV routing, handshake flags, phase-timestamp / peer-metric
  // atomic references) live here; see PdV2RequestContext above.
  PdV2RequestContext pd_v2;

  /*******************************************************************
   * State info used in generation
   * TODO (robertyuan): Move them into a structure later
   *******************************************************************/
  // forwarding_tokens contains tokens used in forwarding step. There are two parts:
  // 1. tokens have kv-caches, kv_cached_token_num is the number
  // 2. tokens need to be processed, their kv-caches are generated during computation
  std::vector<int> forwarding_tokens;

  // tokens generated in current step
  std::vector<int> sampling_result_tokens;

  // Structured generator instance for this request (nullptr if not using structured output)
  std::shared_ptr<StructuredGeneratorInterface> structured_generator = nullptr;

  // Flag indicating whether need to recreate generator on Executor side (after preemption recovery)
  bool need_recreate_generator = false;

  // The intermediate result of beam_search
  std::vector<OutputTuple> &beam_search_group;

  // context decode or decode stage.
  InferStage infer_stage = InferStage::kContext;

  // The decode step, 0 for context decode, and then 1, 2, 3...
  int step = 0;

  // The number of tokens for which kv caches have been generated.
  size_t kv_cached_token_num = 0;

  // The kv cache blocks this request used, shared across devices.
  // The key and value are stored in same blocks.
  std::vector<int> kv_cache_blocks;

  // The SWA kv cache blocks for sliding window attention.
  // swa_kv_cache_blocks[i] holds the memory_block_id of the i-th SWA block.
  // swa_kv_cache_block_idx_offsets[i] is the FA block global index that
  //   swa_kv_cache_blocks[i] corresponds to.
  // Both vectors always have the same length.
  // The covered range must include [max(kv_cached_token_num - sliding_window_size, 0), T].
  std::vector<int> swa_kv_cache_blocks;

  // Per-block FA global index: swa_kv_cache_block_idx_offsets[i] is the FA block
  // global index corresponding to swa_kv_cache_blocks[i].
  // Must have the same length as swa_kv_cache_blocks.
  std::vector<size_t> swa_kv_cache_block_idx_offsets;

  int mamba_kv_cache_block_id = -1;

  // The max token number of one block.
  size_t block_token_num;

  // The offset for model forward's logits output.
  size_t logits_offset = 0;

  // Whether the current req is in pending status of swappiness.
  bool swap_pending = false;

  // The prefix cache tokens number
  int prefix_cache_len = 0;

  // The flexible cache tokens number
  size_t flexible_cache_len = 0;

  // A vector containing pointers to FlexibleCachedCopyTask objects, which represent tasks that involve copying data
  // flexibly between different memory regions.
  std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;

  // The no_repeate ngram sampling map
  NgramDict ngram_dict;

  // The arrive time.
  uint64_t timestamp_in_us;

  // request context
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx;

  // PD 分离：调度完成分发到 executor 前置 true，引用自 Request。
  std::atomic<bool> &prefill_entering_forward;

  // Access-log timestamps, references to Request fields. Stamped via StampOnce().
  std::atomic<uint64_t> &scheduled_time_us;
  std::atomic<uint64_t> &first_token_time_us;
  std::atomic<uint64_t> &finish_time_us;

  // This node's own prefix-cache hit token count (reference to Request field). Single source
  // of truth, written by the scheduler; works even when req_ctx is null (pd_v2 Prefill), so the
  // Prefill node can report it and KsanaPythonOutput / StreamingIterator read it back.
  int &prefix_cache_hit_tokens;

  // Incremental decoded str used in stop strings
  std::string incremental_decoded_str;

  // The number of tokens that have been computed.
  size_t computed_token_num = 0;

  // Whether dispatch infer request in patch mode.
  bool patch_transfer = false;

 public:
  // Init or Recompute, copy output_tokens to prefilling_tokens_
  void ResetPrefillingTokens();

  const std::vector<int> &GetPrefillingTokens() const { return prefilling_tokens_; }
  // 返回 oldest（最先 launched / 最先回流）的 inflight task。
  // 慢路径（深度 1）与原行为一致；build-ahead（深度 2）时即队首 inflight。
  const ScheduleTask &GetInflightTask() const { return inflight_tasks_[0]; }
  const ScheduleTask &GetPlanningTask() const { return planning_task_; }
  const ScheduleTaskWorkload &GetRemainingWorkload() const { return remaining_workload_; }
  const ScheduleTaskWorkload &GetPlanningWorkload() const { return planning_workload_; }

  // 是否还有 inflight task（慢路径等价于队列非空）。
  bool HasInflightTask() const { return inflight_count_ > 0; }
  // 是否还能再容纳一个 inflight task（inflight_count_ < max_depth）。
  // 慢路径传 max_depth=1；build-ahead 传 max_depth=2。
  bool HasInflightCapacity(size_t max_depth) const { return inflight_count_ < max_depth; }
  // 当前 inflight 个数（0/1/2），供 BatchScheduler 流水线深度统计。
  size_t GetInflightCount() const { return inflight_count_; }
  bool HasPlanningTask() const { return !planning_task_.IsEmpty(); }

  // Returns true if the inflight task is a split-fuse prefill step that does NOT cover all remaining
  // prefilling tokens (i.e., there are more prefill chunks to come).
  // 检查 oldest inflight 是否为 split-fuse prefill 中间步；build-ahead 稳态下 prefill 已收敛，通常不触发。
  bool IsSplitPrefillStep() const {
    if (inflight_count_ == 0) {
      return false;
    }
    const auto &wl = inflight_tasks_[0].workload;
    return wl.prefill_token_num > 0 && (wl.prefill_token_num + wl.prefill_start_offset < prefilling_tokens_.size());
  }

  void SetInflightTaskGenResultEstimation(size_t generated_token_num, size_t draft_token_num);

  void SetRemainingWorkload(const ScheduleTaskWorkload &workload);
  void SetPlanningWorkload(const ScheduleTaskWorkload &workload);
  void SetPlanningTask();

  // retire 队首（oldest）inflight task，FIFO 滑动后续 task；慢路径等价于清空唯一 inflight。
  void ResetInflightTask();

  // 清掉 swap-out 前设置但尚未 launch 的 planning_task_。否则 swap-in 后下一轮调度会因为
  // HasPlanningTask()=true 一直跳过该请求。
  void ResetPlanningTask() { planning_task_.Reset(); }

  // 提前释放 chunked 请求的 inflight slot, 让下一 pp_batch 立即 launch 后续 chunk.
  void RetireChunkedReqs();

  // 返回 false 表示本次 sample 回流被 RetireChunkedReqs 提前处理过, caller 不应再 ResetInflightTask.
  bool UpdateAfterInflightTaskFinished();

  bool IsEosGenerated() const { return is_eos_generated_; }

  // 把当前 planning_task_ launch 为 inflight；按 inflight_count_ 写入对应 slot。
  // use_placeholder_for_last_token=true（build-ahead 深度 2 的第二次 launch）：末位 generated_token 用占位符，
  // 位置记入 pending_placeholder_positions_，oldest retire 时回填真实值。慢路径不传第二参数即可。
  void LaunchPlanningTask(bool use_placeholder_for_last_token = false);

  // forwarding_tokens 末位占位 token id（-1，便于与合法 vocab id 区分）。Executor ReplaceLastTokenKernel 必然覆盖。
  // 复用 infer_stage.h 共享常量，Engine / Executor 两侧一致。
  static constexpr int kFastPathPlaceholderTokenId = ::ksana_llm::kFastPathPlaceholderTokenId;
  // 同时存在 inflight 的硬上限，与 BatchScheduler build-ahead 的 max_depth=2 对齐。
  // FastPathController 启用 → 深度 2；禁用 → 实际只用 1。
  static constexpr size_t kMaxInflightDepth = 2;

  // 将全局 cache 命中的生成 token(s) 追加到 output_tokens 和 prefilling_tokens_，
  // 供全命中后直接进入 decode 时使用（支持 MTP 多 token）。
  void AppendCachedTokens(const std::vector<int> &tokens);

  std::string ScheduleStateToStr() const;

  bool IsStopped() const { return is_stopped_; }
  void Stop() { is_stopped_ = true; }
  void Restart() { is_stopped_ = false; }

 private:
  bool is_stopped_ = false;
  std::vector<int> prefilling_tokens_;

  // inflight tasks 队列（深度上限 kMaxInflightDepth=2）；慢路径只用 [0]。
  // [0]=oldest（最先 launched/retire），[1]=newest（深度 2 时存在）。
  // 与原 inflight_task_ 字段等价：GetInflightTask() 返回 [0]，ResetInflightTask() 弹出 [0] 并滑动 [1]。
  std::array<ScheduleTask, kMaxInflightDepth> inflight_tasks_;
  size_t inflight_count_ = 0;
  // 兼容字段（历史声明，暂未使用）；保留以便后续估算分摊到 oldest inflight。
  size_t inflight_task_estimated_generated_token_num_ = 0;
  size_t inflight_task_estimated_draft_token_num_ = 0;

  // 与 placeholder launch 的 inflight task 对应的 forwarding_tokens 末位下标（FIFO，大小 <= 1）。
  // UpdateAfterInflightTaskFinished 在 oldest retire 时用 generated_tokens 回填。慢路径为空 deque，零开销。
  std::deque<size_t> pending_placeholder_positions_;

  // RetireChunkedReqs 已 reset inflight, 后续 UpdateAfterInflightTaskFinished 应跳过正常处理.
  size_t inflight_update_skip_count_ = 0;

  ScheduleTask planning_task_;
  ScheduleTaskWorkload planning_workload_;
  ScheduleTaskWorkload remaining_workload_;
  bool is_eos_generated_ = false;

  // admission 阶段注册的 teardown 回调，析构时统一触发，不依赖请求是完成、abort 还是 cancel。
  std::vector<std::function<void()>> on_destroy_callbacks_;
};

inline std::string RequestVector2Str(const std::vector<std::shared_ptr<InferRequest>> &infer_requests) {
  std::ostringstream ss;
  ss << "[";
  for (const auto &req : infer_requests) {
    if (req != nullptr) {
      ss << req->req_id << ",";
    }
  }
  ss << "]";
  return ss.str();
}

}  // namespace ksana_llm
