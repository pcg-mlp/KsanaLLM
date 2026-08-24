/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "endpoints/streaming/streaming_iterator.h"

#include <thread>

#include "runtime/infer_request.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

static std::unordered_map<int64_t, std::shared_ptr<DyingRequest>> dying_requests;
static std::mutex dying_mutex;

StreamingIterator::~StreamingIterator() {
  // Process dying only if request is not finished.
  if (!all_finished) {
    KLLM_LOG_DEBUG << "StreamingIterator client disconnected, req " << request_->req_id;

    std::shared_ptr<DyingRequest> dying_req = std::make_shared<DyingRequest>();
    dying_req->request = request_;
    dying_req->ksana_python_input = ksana_python_input_;

    int64_t req_id = request_->req_id;

    {
      std::lock_guard<std::mutex> guard(dying_mutex);
      dying_requests[req_id] = dying_req;
    }

    std::thread([req_id]() {
      std::shared_ptr<DyingRequest> dr = nullptr;
      {
        std::lock_guard<std::mutex> guard(dying_mutex);
        auto it = dying_requests.find(req_id);
        if (it == dying_requests.end()) {
          return;
        }
        dr = it->second;
      }

      // Delay exist if client is aborted.
      dr->request->aborted = true;
      dr->request->abort_waiter->Wait();

      {
        KLLM_LOG_DEBUG << "StreamingIterator disconnected req " << req_id << " finished.";
        std::lock_guard<std::mutex> guard(dying_mutex);
        dying_requests.erase(req_id);
      }

      // Import: Make sure the destroying of ksana_python_input is in python GIL.
      // Otherwise it maybe cause crash when pyobject in ksana_python_input destroyed.
      {
        pybind11::gil_scoped_acquire acquire;
        dr = nullptr;
      }
    }).detach();
  }
}

bool StreamingIterator::AddOutput(ksana_llm::KsanaPythonOutput& ksana_python_output) {
  size_t total_token_nums = 0;
  for (size_t i = 0; i < request_->output_group.size(); i++) {
    OutputTuple& output = request_->output_group[i];
    total_token_nums += std::get<0>(output).size();
  }

  if (!request_->has_stop_strings && total_token_nums == total_token_nums_) {
    return false;
  }

  total_token_nums_ = total_token_nums;
  ksana_python_output.input_tokens = request_->input_tokens;

  // Transfer prefix_cache_hit_tokens only once (read from the Request member, the single
  // source of truth the scheduler writes).
  if (!prefix_cache_hit_tokens_read_) {
    prefix_cache_hit_tokens_read_ = true;
    prefix_cache_hit_tokens_ = request_->prefix_cache_hit_tokens;
  }
  ksana_python_output.prefix_cache_hit_tokens = prefix_cache_hit_tokens_;

  for (size_t i = 0; i < request_->output_group.size(); i++) {
    OutputTuple& output = request_->output_group[i];
    const auto& output_tokens = std::get<0>(output);
    ksana_python_output.output_tokens.emplace_back(output_tokens.begin() + request_->input_tokens.size(),
                                                   output_tokens.end());
    if (return_logprobs_) {
      ksana_python_output.logprobs.push_back(std::get<1>(output));
    }
  }
  if (!request_->cache_stat.empty()) {
    ksana_python_output.cache_stat = std::move(request_->cache_stat);
  }

  // Snapshot access-log timestamps so Python can compute waiting/ttft/tpot/total.
  bool all_reqs_finished = !request_->finisheds.empty();
  for (bool req_finished : request_->finisheds) {
    all_reqs_finished = all_reqs_finished && req_finished;
  }
  SnapshotTimestamps(ksana_python_output, all_reqs_finished);
  return true;
}

void StreamingIterator::SnapshotTimestamps(ksana_llm::KsanaPythonOutput& ksana_python_output,
                                           bool wait_put_done) const {
  if (wait_put_done) {
    request_->WaitGlobalCachePutDone();
  }

  ksana_python_output.arrive_time_us = request_->timestamp_in_us;
  ksana_python_output.scheduled_time_us = request_->scheduled_time_us.load();
  ksana_python_output.first_token_time_us = request_->first_token_time_us.load();
  ksana_python_output.finish_time_us = request_->finish_time_us.load();

  // pd_v2 Decode-side phase timestamps + peer-Prefill metrics for the access log.
  ksana_python_output.pd_v2_blocks_allocated_time_us = request_->pd_v2_blocks_allocated_time_us.load();
  ksana_python_output.pd_v2_prefill_dispatched_time_us = request_->pd_v2_prefill_dispatched_time_us.load();
  ksana_python_output.pd_v2_prefill_complete_time_us = request_->pd_v2_prefill_complete_time_us.load();
  ksana_python_output.pd_v2_pf_prefix_cache_hit_tokens = request_->pd_v2_pf_prefix_cache_hit_tokens.load();
  ksana_python_output.pd_v2_pf_queue_us = request_->pd_v2_pf_queue_us.load();
  ksana_python_output.pd_v2_pf_compute_us = request_->pd_v2_pf_compute_us.load();
  ksana_python_output.pd_v2_pf_send_us = request_->pd_v2_pf_send_us.load();
  if (request_->req_ctx && request_->req_ctx->count("pd_v2_prefill_peer")) {
    ksana_python_output.pd_v2_prefill_peer = request_->req_ctx->at("pd_v2_prefill_peer");
  }

  // Prefix-cache hit from the Request member (single source of truth). Draft stats still come
  // from the req_ctx side-channel (written by the runtime per accepted draft batch).
  ksana_python_output.prefix_cache_hit_tokens = request_->prefix_cache_hit_tokens;
  if (request_->req_ctx) {
    if (request_->req_ctx->count("mtp_hit_tokens")) {
      ksana_python_output.draft_hit_tokens = std::stoi(request_->req_ctx->at("mtp_hit_tokens"));
    }
    if (request_->req_ctx->count("mtp_draft_tokens")) {
      ksana_python_output.draft_tokens = std::stoi(request_->req_ctx->at("mtp_draft_tokens"));
    }
  }

  ksana_python_output.global_cache_ext_tokens = request_->global_cache_ext_tokens.load(std::memory_order_relaxed);
  ksana_python_output.global_cache_get_us = request_->global_cache_get_us.load(std::memory_order_relaxed);
  ksana_python_output.global_cache_put_fa_blocks = request_->global_cache_put_fa_blocks.load(std::memory_order_relaxed);
  ksana_python_output.global_cache_put_swa_blocks =
      request_->global_cache_put_swa_blocks.load(std::memory_order_relaxed);
  const uint64_t put_done = request_->global_cache_put_done_time_us.load(std::memory_order_relaxed);
  const uint64_t finish = ksana_python_output.finish_time_us;
  if (request_->global_cache_put_pending.load(std::memory_order_acquire)) {
    ksana_python_output.global_cache_put_after_finish_ms = -1.0;
  } else if (put_done > 0 && finish > 0 && put_done > finish) {
    ksana_python_output.global_cache_put_after_finish_ms = static_cast<double>(put_done - finish) / 1000.0;
  } else {
    ksana_python_output.global_cache_put_after_finish_ms = 0.0;
  }
}

Status StreamingIterator::GetNext(ksana_llm::KsanaPythonOutput& ksana_python_output) {
  while (true) {
    if (request_->aborted) {
      // Do NOT set all_finished=true here. The ~StreamingIterator() dying_request
      // keep-alive flow only runs when !all_finished. Without it, ksana_python_input_
      // is destroyed immediately, but Request holds bare references into it (logprobs,
      // output_tokens, request_target, etc — see request.h). The scheduler's ModelDriver
      // thread is still processing the request and will UAF on those references.
      return Status(RET_REQUEST_TERMINATED, "client aborted");
    }
    if (all_finished) {
      // failure, no token generated.
      if (!request_->finish_status.OK()) {
        return request_->finish_status;
      }
      return Status(RET_STOP_ITERATION);
    }
    // Waiting util next token generated.
    request_->step_waiter->Wait();
    request_->step_waiter->Reset(1);

    // PD 分离：调度完成即将进入 Forward，返回一个空 chunk 作为 prefill 通知。
    // Router 按位置（第一个 chunk）识别通知，无需特殊字段。
    // exchange(false) 保证只触发一次。
    if (request_->prefill_entering_forward.exchange(false)) {
      return Status();
    }

    all_finished = true;
    for (bool req_finished : request_->finisheds) {
      all_finished = all_finished && req_finished;
    }
    // Defensive: if a scheduler path notifies before stamping finish_time_us,
    // fall back to stamping here so Python's SnapshotTimestamps never reads 0.
    // StampOnce is idempotent — scheduler's stamp still wins when it ran first.
    if (all_finished) {
      StampOnce(request_->finish_time_us);
    }
    {
      // Fetch next token util the last token is fetched.
      std::unique_lock<std::mutex> lock(request_->output_mutex);
      if (!AddOutput(ksana_python_output)) continue;
    }
    return Status();
  }
  return Status();
}

void StreamingIterator::Abort() {
  // Idempotent: safe to call multiple times. Wakes step_waiter so any in-flight
  // GetNext() returns immediately with RET_REQUEST_TERMINATED. Setting `aborted`
  // also makes the scheduler StopRequest the request on its next sweep
  // (continuous_batching.cpp running/waiting queue checks).
  if (all_finished) {
    return;
  }
  request_->aborted = true;
  if (request_->step_waiter) {
    request_->step_waiter->Stop();
  }
}

}  // namespace ksana_llm
