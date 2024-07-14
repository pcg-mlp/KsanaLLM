/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler.h"
#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler_test_helper.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/threadpool.h"

#include "ksana_llm/profiler/timer.h"

#include <unordered_map>
#include <sstream>

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

class ParallelTester {
 public:
  ParallelTester(BatchScheduler* batch_scheduler, BatchSchedulerEvironmentSimulator* env_simulator)
      : batch_scheduler_(batch_scheduler), env_simulator_(env_simulator) {}

  struct RequestInfo {
    int req_id;
    int expect_output_token_num;
    int input_token_num;
    std::shared_ptr<Request> req;
    std::vector<std::shared_ptr<InferRequest>> infer_req_group;
  };

  // hooks used during execution.
  class ExeHookInterface {
   public:
    virtual void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) {}

    virtual void CheckRequestsAfterExecution(const std::vector<RequestInfo>& reqs) {}

   public:
    int before_step_num = 0;
    int after_exe_num = 0;
  };

  // This hook checks results when  all requests are finished as expected.
  class DefaultResultCheckHook : public ExeHookInterface {
   public:
    DefaultResultCheckHook(BatchSchedulerEvironmentSimulator* env_simulator) : env_simulator_(env_simulator) {}
    ~DefaultResultCheckHook(){
      NLLM_LOG_INFO << "~DefaultResultCheckHook, after_exe_num=" << after_exe_num;
      EXPECT_GT(after_exe_num, 0); // CheckRequestsAfterExecution must be invoked. Maybe this hook is not added to hook list.
    }

    void CheckRequestsAfterExecution(const std::vector<RequestInfo>& reqs) override {
      after_exe_num++;
      for (auto& info : reqs) {
        for (auto& infer_req : info.infer_req_group) {
          env_simulator_->CheckRequestOutput(infer_req);
        }
      }
    }

   private:
    BatchSchedulerEvironmentSimulator* env_simulator_;
  };

  void InitRequestInfoListByDefault(std::vector<RequestInfo>& reqs) {
    std::vector<std::pair<int, int>> seeds;
    seeds.resize(1);
    seeds[0].first = 0;
    for (auto& info : reqs) {
      seeds[0].second = info.req_id;
      info.infer_req_group =
          env_simulator_->InitRequest(info.req_id, info.input_token_num, info.expect_output_token_num, info.req, seeds);
    }
  }

  void DoParallelRequestAndCheck(int client_num, std::vector<RequestInfo>& reqs, std::vector<ExeHookInterface*>& hooks,
                                 int timeout = 5) {
    NLLM_LOG_INFO << "DoParallelRequestAndCheck start.  client_num=" << client_num << ", request_num=" << reqs.size();
    NLLM_CHECK_WITH_INFO(hooks.size() > 0, "There must be some hooks");

    time_t start_time = ProfileTimer::GetCurrentTime();
    ClientSimulator client_simulator(client_num, batch_scheduler_);
    for (auto& info : reqs) {
      client_simulator.AddInferRequests(info.req_id, info.infer_req_group);
    }

    // Wait for request enqueue
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    // schedule and generate tokens
    while (true) {
      std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;
      scheduled_reqs = batch_scheduler_->Schedule();
      if (scheduled_reqs.empty()) {
        if (client_simulator.IsAllRequestFinished()) {
          NLLM_LOG_INFO << "All requests finished";
          break;
        }
        time_t cur_time = ProfileTimer::GetCurrentTime();
        if ((cur_time - start_time) > timeout) {
          NLLM_LOG_INFO << "Test Timeout. timeout=" << timeout << " seconds";
          break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        continue;
      }

      for (auto hook : hooks) {
        hook->CheckRequestsBeforeAStep(scheduled_reqs);
      }
      env_simulator_->RunAStep(scheduled_reqs);
    }

    // Check request results
    for (auto hook : hooks) {
      hook->CheckRequestsAfterExecution(reqs);
    }
    NLLM_LOG_INFO << "DoParallelRequestAndCheck finished";
  }

 private:
  BatchScheduler* batch_scheduler_;
  BatchSchedulerEvironmentSimulator* env_simulator_;
};

class PrintStepHook : public ParallelTester::ExeHookInterface {
 public:
  PrintStepHook(bool print_all_blocks = false) : print_all_blocks_(print_all_blocks) {}
  ~PrintStepHook(){
    NLLM_LOG_INFO << "~PrintStepHook, before_step_num=" << before_step_num;
    EXPECT_GT(before_step_num, 0);
  }

  void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) override {
    before_step_num++;
    for (auto& req : reqs) {
      std::ostringstream ss;
      ss << "Step " << before_step_num << ": req_id:" << req->req_id
         << ", output_tokens.size()=" << req->output_tokens.size();
      if (print_all_blocks_) {
        ss << ", blocks={ ";
        for (size_t i = 0; i < req->kv_cache_blocks.size(); i++) {
          auto& blocks = req->kv_cache_blocks[i];
          ss << i << "={ ";
          for (auto blk_id : blocks) {
            ss << blk_id << ", ";
          }
        }
      }
      ss << "} ";
      NLLM_LOG_INFO << ss.str(); 
    }
  }

 private:
  bool print_all_blocks_;
};

}  // namespace ksana_llm
