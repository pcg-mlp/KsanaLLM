/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "endpoints/streaming/streaming_iterator.h"

#include <thread>

#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

static std::unordered_map<int, std::shared_ptr<DyingRequest>> dying_requests;
static std::mutex dying_mutex;

StreamingIterator::~StreamingIterator() {
  // Process dying only if request is not finished.
  if (!all_finished) {
    KLLM_LOG_DEBUG << "StreamingIterator client disconnected, req " << request_->req_id;

    std::shared_ptr<DyingRequest> dying_req = std::make_shared<DyingRequest>();
    dying_req->request = request_;
    dying_req->ksana_python_input = ksana_python_input_;

    int req_id = request_->req_id;

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

  // Transfer prefix_cache_hit_tokens only once
  if (!prefix_cache_hit_tokens_read_) {
    prefix_cache_hit_tokens_read_ = true;
    if (request_->req_ctx && request_->req_ctx->count("prefix_cache_hit_tokens")) {
      prefix_cache_hit_tokens_ = std::stoi(request_->req_ctx->at("prefix_cache_hit_tokens"));
    }
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
  return true;
}

Status StreamingIterator::GetNext(ksana_llm::KsanaPythonOutput& ksana_python_output) {
  while (true) {
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
    {
      // Fetch next token util the last token is fetched.
      std::unique_lock<std::mutex> lock(request_->output_mutex);
      if (!AddOutput(ksana_python_output)) continue;
    }
    return Status();
  }
  return Status();
}

}  // namespace ksana_llm
