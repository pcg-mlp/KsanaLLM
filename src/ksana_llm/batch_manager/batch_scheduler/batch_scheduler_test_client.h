/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/threadpool.h"

#include <unordered_map>

namespace ksana_llm {

class ClientSimulator {
 public:
  ClientSimulator(int thread_num, BatchScheduler* batch_scheduler)
      : scheduler_(batch_scheduler), thread_num_(thread_num), threadpool_(thread_num), is_destroying_(false) {
    threadpool_.Start();
  }

  ~ClientSimulator() {
    NLLM_LOG_INFO << "~ClientSimulator IsAllRequestFinished=" << IsAllRequestFinished();
    is_destroying_ = true;
    if (!IsAllRequestFinished()) {
      for (auto& it : client_req_map_) {
        it.second.req_group[0]->waiter->Notify();
        NLLM_LOG_INFO << "Notify unfinished req: " << it.second.req_group[0]->req_id;
      }
    }
    threadpool_.Stop();
  }

  void AddInferRequests(int req_group_id, std::vector<std::shared_ptr<InferRequest>>& infer_reqs) {
    std::unordered_map<int, ClientRequest>::iterator iter;
    NLLM_CHECK_WITH_INFO(infer_reqs.size() > 0, FormatStr("infer_reqs.size()==%d, must >0.", infer_reqs.size()));
    {
      std::lock_guard<std::mutex> guard(mux_);
      NLLM_CHECK_WITH_INFO(client_req_map_.find(req_group_id) == client_req_map_.end(),
                           FormatStr("req_group_id %d exists.", req_group_id));
      ClientRequest dummy_req;
      client_req_map_[req_group_id] = dummy_req;
      iter = client_req_map_.find(req_group_id);
      iter->second.req_group = infer_reqs;
    }
    threadpool_.Submit([=]() -> int {
      if (is_destroying_) {
        return 0;
      }
      ClientRequest& req = iter->second;
      req.enqueue_status = scheduler_->AddInferRequest(req.req_group);
      // all requests in req_group come from same request.
      req.req_group[0]->waiter->Wait();
      req.is_finished = true;
      return 0;
    });
  }

  void AddAnInferRequest(std::shared_ptr<InferRequest>& infer_req) {
    std::vector<std::shared_ptr<InferRequest>> reqs;
    reqs.push_back(infer_req);
    AddInferRequests(infer_req->req_id, reqs);
  }

  bool IsAllRequestFinished() {
    for (auto& it : client_req_map_) {
      if (!it.second.is_finished) {
        return false;
      }
    }
    return true;
  }

 private:
  struct ClientRequest {
    std::vector<std::shared_ptr<InferRequest>> req_group;
    bool is_finished = false;
    Status enqueue_status;
  };

  BatchScheduler* scheduler_;
  int thread_num_;
  ThreadPool threadpool_;
  bool is_destroying_;

  std::unordered_map<int, ClientRequest> client_req_map_;
  std::mutex mux_;
};

}  // namespace ksana_llm
