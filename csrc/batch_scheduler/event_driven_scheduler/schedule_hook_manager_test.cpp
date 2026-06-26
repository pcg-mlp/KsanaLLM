/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================
 */

#include <algorithm>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager.h"
#include "runtime/infer_request.h"
#include "tests/test.h"
#include "utils/request.h"

namespace ksana_llm {
namespace {

std::shared_ptr<InferRequest> CreateInferRequest(int64_t req_id, std::shared_ptr<KsanaPythonInput>* input_holder,
                                                 std::shared_ptr<Request>* request_holder, uint32_t dp_idx = 0) {
  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = req_id;
  infer_req->attn_dp_group_id = dp_idx;
  *input_holder = std::move(input);
  *request_holder = std::move(request);
  return infer_req;
}

class MockScheduleHook : public ScheduleHookInterface {
 public:
  using ProcessFn = std::function<ReqIdToNextStep(const std::vector<InferRequestPtr>&, ScheduleContext&)>;
  using UpdateFn = std::function<ReqIdToNextStep(const std::vector<AsyncTaskResult>&, ScheduleContext&)>;

  explicit MockScheduleHook(ProcessFn process_fn = nullptr, UpdateFn update_fn = nullptr)
      : process_fn_(std::move(process_fn)), update_fn_(std::move(update_fn)) {}

  ReqIdToNextStep Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) override {
    process_called_req_ids_.push_back(GetReqIds(requests));
    if (process_fn_ != nullptr) {
      return process_fn_(requests, context);
    }
    return {};
  }

  ReqIdToNextStep UpdateAsyncResult(const std::vector<AsyncTaskResult>& finished_tasks,
                                    ScheduleContext& context) override {
    update_called_task_hook_idxs_.push_back(GetHookIdxs(finished_tasks));
    if (update_fn_ != nullptr) {
      return update_fn_(finished_tasks, context);
    }
    return {};
  }

  void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) override {
    callback_ = std::move(callback);
    ++set_callback_count_;
  }

  void TriggerAsyncTaskFinished() {
    if (callback_ != nullptr) {
      EventAsyncTaskFinishedInfo info;
      callback_(info);
    }
  }

  const std::vector<std::vector<int64_t>>& ProcessCalledReqIds() const { return process_called_req_ids_; }
  const std::vector<std::vector<int>>& UpdateCalledTaskHookIdxs() const { return update_called_task_hook_idxs_; }
  int SetCallbackCount() const { return set_callback_count_; }

 private:
  static std::vector<int64_t> GetReqIds(const std::vector<InferRequestPtr>& requests) {
    std::vector<int64_t> ids;
    ids.reserve(requests.size());
    for (const auto& req : requests) {
      ids.push_back(req->req_id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
  }

  static std::vector<int> GetHookIdxs(const std::vector<AsyncTaskResult>& finished_tasks) {
    std::vector<int> hook_idxs;
    hook_idxs.reserve(finished_tasks.size());
    for (const auto& task : finished_tasks) {
      hook_idxs.push_back(task.hook_idx);
    }
    std::sort(hook_idxs.begin(), hook_idxs.end());
    return hook_idxs;
  }

 private:
  ProcessFn process_fn_;
  UpdateFn update_fn_;
  AsyncTaskFinishedCallback callback_;
  int set_callback_count_ = 0;
  std::vector<std::vector<int64_t>> process_called_req_ids_;
  std::vector<std::vector<int>> update_called_task_hook_idxs_;
};

TEST(ScheduleHookManagerTest, SetAsyncTaskFinishedCallbackAppliedToHooks) {
  ScheduleHookManager manager;
  int callback_called_count = 0;
  manager.SetAsyncTaskFinishedCallback(
      [&callback_called_count](const EventAsyncTaskFinishedInfo&) { ++callback_called_count; });

  auto hook1 = std::make_shared<MockScheduleHook>();
  auto hook2 = std::make_shared<MockScheduleHook>();

  manager.AddHook(nullptr);
  manager.AddHook(hook1);
  manager.AddHook(hook2);

  EXPECT_EQ(hook1->SetCallbackCount(), 1);
  EXPECT_EQ(hook2->SetCallbackCount(), 1);

  hook1->TriggerAsyncTaskFinished();
  hook2->TriggerAsyncTaskFinished();
  EXPECT_EQ(callback_called_count, 2);

  manager.SetAsyncTaskFinishedCallback(
      [&callback_called_count](const EventAsyncTaskFinishedInfo&) { callback_called_count += 10; });
  EXPECT_EQ(hook1->SetCallbackCount(), 2);
  EXPECT_EQ(hook2->SetCallbackCount(), 2);
  hook1->TriggerAsyncTaskFinished();
  EXPECT_EQ(callback_called_count, 12);
}

TEST(ScheduleHookManagerTest, ProcessFollowsChainAndCollectsFinalNextStep) {
  ScheduleHookManager manager;
  std::vector<std::shared_ptr<KsanaPythonInput>> input_holders;
  std::vector<std::shared_ptr<Request>> request_holders;
  std::shared_ptr<KsanaPythonInput> input_holder;
  std::shared_ptr<Request> request_holder;
  auto req1 = CreateInferRequest(1, &input_holder, &request_holder);
  input_holders.push_back(std::move(input_holder));
  request_holders.push_back(std::move(request_holder));
  auto req2 = CreateInferRequest(2, &input_holder, &request_holder);
  input_holders.push_back(std::move(input_holder));
  request_holders.push_back(std::move(request_holder));
  auto req3 = CreateInferRequest(3, &input_holder, &request_holder);
  input_holders.push_back(std::move(input_holder));
  request_holders.push_back(std::move(request_holder));
  std::vector<ScheduleHookInterface::InferRequestPtr> requests = {req1, req2, req3};
  ScheduleContext context;

  auto hook0 = std::make_shared<MockScheduleHook>(
      [](const std::vector<ScheduleHookInterface::InferRequestPtr>&, ScheduleContext&) {
        ReqIdToNextStep next;
        next[1] = NextStep{NextStepAction::kWaitAsyncResult, 101};
        next[2] = NextStep{NextStepAction::kContinue, 0};
        return next;
      });

  auto hook1 = std::make_shared<MockScheduleHook>(
      [](const std::vector<ScheduleHookInterface::InferRequestPtr>&, ScheduleContext&) {
        ReqIdToNextStep next;
        next[2] = NextStep{NextStepAction::kFailAndEnd, 0};
        next[3] = NextStep{NextStepAction::kContinue, 0};
        return next;
      });

  auto hook2 = std::make_shared<MockScheduleHook>(
      [](const std::vector<ScheduleHookInterface::InferRequestPtr>&, ScheduleContext&) {
        ReqIdToNextStep next;
        next[3] = NextStep{static_cast<NextStepAction>(999), 0};
        return next;
      });

  manager.AddHook(hook0);
  manager.AddHook(hook1);
  manager.AddHook(hook2);

  ReqIdToNextStep result = manager.Process(requests, context);
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[1].action, NextStepAction::kWaitAsyncResult);
  EXPECT_EQ(result[1].task_id, 101);
  EXPECT_EQ(result[2].action, NextStepAction::kFailAndEnd);
  EXPECT_EQ(result[3].action, NextStepAction::kFailAndEnd);

  ASSERT_EQ(hook0->ProcessCalledReqIds().size(), 1u);
  ASSERT_EQ(hook1->ProcessCalledReqIds().size(), 1u);
  ASSERT_EQ(hook2->ProcessCalledReqIds().size(), 1u);
  EXPECT_EQ(hook0->ProcessCalledReqIds()[0], (std::vector<int64_t>{1, 2, 3}));
  EXPECT_EQ(hook1->ProcessCalledReqIds()[0], (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(hook2->ProcessCalledReqIds()[0], (std::vector<int64_t>{3}));
}

TEST(ScheduleHookManagerTest, ProcessWithoutHooksReturnsContinue) {
  ScheduleHookManager manager;
  ScheduleContext context;
  std::vector<std::shared_ptr<KsanaPythonInput>> input_holders;
  std::vector<std::shared_ptr<Request>> request_holders;
  std::shared_ptr<KsanaPythonInput> input_holder;
  std::shared_ptr<Request> request_holder;
  auto req1 = CreateInferRequest(100, &input_holder, &request_holder);
  input_holders.push_back(std::move(input_holder));
  request_holders.push_back(std::move(request_holder));
  auto req2 = CreateInferRequest(200, &input_holder, &request_holder);
  input_holders.push_back(std::move(input_holder));
  request_holders.push_back(std::move(request_holder));
  std::vector<ScheduleHookInterface::InferRequestPtr> requests = {req1, req2};

  ReqIdToNextStep result = manager.Process(requests, context);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[100].action, NextStepAction::kContinue);
  EXPECT_EQ(result[200].action, NextStepAction::kContinue);
}

TEST(ScheduleHookManagerTest, UpdateAsyncResultContinuesFromCorrectHookIndex) {
  ScheduleHookManager manager;

  std::vector<std::shared_ptr<KsanaPythonInput>> input_holders;
  std::vector<std::shared_ptr<Request>> request_holders;
  std::shared_ptr<KsanaPythonInput> input_holder;
  std::shared_ptr<Request> request_holder;
  auto req1 = CreateInferRequest(1, &input_holder, &request_holder, 0);
  input_holders.push_back(std::move(input_holder));
  request_holders.push_back(std::move(request_holder));
  auto req2 = CreateInferRequest(2, &input_holder, &request_holder, 0);
  input_holders.push_back(std::move(input_holder));
  request_holders.push_back(std::move(request_holder));
  auto req3 = CreateInferRequest(3, &input_holder, &request_holder, 0);
  input_holders.push_back(std::move(input_holder));
  request_holders.push_back(std::move(request_holder));

  ScheduleState schedule_state;
  schedule_state.schedule_states.resize(1);
  schedule_state.schedule_states[0].resize(1);
  schedule_state.req_id_to_req_ptr[1] = req1;
  schedule_state.req_id_to_req_ptr[2] = req2;
  schedule_state.req_id_to_req_ptr[3] = req3;
  schedule_state.req_to_batch_idx[1] = 0;
  schedule_state.req_to_batch_idx[2] = 0;
  schedule_state.req_to_batch_idx[3] = 0;
  schedule_state.schedule_states[0][0].preprocessing_reqs[1] = req1;
  schedule_state.schedule_states[0][0].preprocessing_reqs[2] = req2;
  schedule_state.schedule_states[0][0].preprocessing_reqs[3] = req3;

  ScheduleContext context;
  context.SetScheduleState(&schedule_state);

  auto hook0 = std::make_shared<MockScheduleHook>(nullptr, [](const std::vector<AsyncTaskResult>&, ScheduleContext&) {
    ReqIdToNextStep next;
    next[1] = NextStep{NextStepAction::kContinue, 0};
    next[2] = NextStep{NextStepAction::kWaitAsyncResult, 202};
    next[999] = NextStep{NextStepAction::kContinue, 0};
    return next;
  });

  auto hook1 = std::make_shared<MockScheduleHook>(
      [](const std::vector<ScheduleHookInterface::InferRequestPtr>& requests, ScheduleContext&) {
        ReqIdToNextStep next;
        if (!requests.empty()) {
          next[1] = NextStep{NextStepAction::kContinue, 0};
        }
        return next;
      },
      [](const std::vector<AsyncTaskResult>&, ScheduleContext&) {
        ReqIdToNextStep next;
        next[3] = NextStep{NextStepAction::kContinue, 0};
        return next;
      });

  auto hook2 = std::make_shared<MockScheduleHook>(
      [](const std::vector<ScheduleHookInterface::InferRequestPtr>& requests, ScheduleContext&) {
        ReqIdToNextStep next;
        for (const auto& req : requests) {
          if (req->req_id == 1) {
            next[1] = NextStep{NextStepAction::kFailAndEnd, 0};
          } else if (req->req_id == 3) {
            next[3] = NextStep{NextStepAction::kContinue, 0};
          }
        }
        return next;
      });

  manager.AddHook(hook0);
  manager.AddHook(hook1);
  manager.AddHook(hook2);

  std::vector<AsyncTaskResult> finished_tasks = {AsyncTaskResult{0, 10, 0}, AsyncTaskResult{1, 11, 0},
                                                 AsyncTaskResult{2, 12, 0}};
  ReqIdToNextStep result = manager.UpdateAsyncResult(finished_tasks, context);

  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[1].action, NextStepAction::kFailAndEnd);
  EXPECT_EQ(result[2].action, NextStepAction::kWaitAsyncResult);
  EXPECT_EQ(result[2].task_id, 202);
  EXPECT_EQ(result[3].action, NextStepAction::kContinue);

  ASSERT_EQ(hook0->UpdateCalledTaskHookIdxs().size(), 1u);
  ASSERT_EQ(hook1->UpdateCalledTaskHookIdxs().size(), 1u);
  ASSERT_EQ(hook2->UpdateCalledTaskHookIdxs().size(), 1u);
  EXPECT_EQ(hook0->UpdateCalledTaskHookIdxs()[0], (std::vector<int>{0}));
  EXPECT_EQ(hook1->UpdateCalledTaskHookIdxs()[0], (std::vector<int>{1}));
  EXPECT_EQ(hook2->UpdateCalledTaskHookIdxs()[0], (std::vector<int>{2}));

  ASSERT_EQ(hook1->ProcessCalledReqIds().size(), 1u);
  ASSERT_EQ(hook2->ProcessCalledReqIds().size(), 1u);
  EXPECT_EQ(hook1->ProcessCalledReqIds()[0], (std::vector<int64_t>{1}));
  EXPECT_EQ(hook2->ProcessCalledReqIds()[0], (std::vector<int64_t>{1, 3}));
}

}  // namespace
}  // namespace ksana_llm
