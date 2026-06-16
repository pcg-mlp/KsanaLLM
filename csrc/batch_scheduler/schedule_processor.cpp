/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/schedule_processor.h"

#include "batch_scheduler/schedule_output_process.h"
#include "runtime/infer_request.h"
#include "utils/logger.h"

namespace ksana_llm {

void ScheduleProcessor::Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler) {
  batch_scheduler_ = batch_scheduler;
  if (enable_async_) {
    planning_sched_results_.resize(max_pp_batch_num_, nullptr);
    for (size_t multi_batch_id = 0; multi_batch_id < max_pp_batch_num_; ++multi_batch_id) {
      sched_threads_.emplace_back(&ScheduleProcessor::AsyncScheduleThread, this, multi_batch_id);
    }
  } else {
    sync_generating_finished_.resize(max_pp_batch_num_, true);
    if (max_pp_batch_num_ > 1) {
      for (size_t multi_batch_id = 0; multi_batch_id < max_pp_batch_num_; ++multi_batch_id) {
        sched_threads_.emplace_back(&ScheduleProcessor::SyncScheduleThread, this, multi_batch_id);
      }
    }
  }

  KLLM_LOG_INFO << "ScheduleProcessor initialized";
}

Status ScheduleProcessor::AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  return batch_scheduler_->AddInferRequest(infer_request_group);
}

// The sync mode: call Schedule directly -> check for running requests -> wait or process data
std::shared_ptr<ScheduleResult> ScheduleProcessor::GetNextScheduleResult() {
  if (enable_async_ || max_pp_batch_num_ > 1) {
    return sched_result_queue_.Get();
  }
  return GetScheduleResultInSync(0);
}

void ScheduleProcessor::UpdateWithGenerationResult(const GenerationOutputGroup& generation_output) {
  size_t multi_batch_id = generation_output.multi_batch_id;
  batch_scheduler_->UpdateWithGenerationResult(multi_batch_id, generation_output);

  // No need to lock here because only one thread will call UpdateWithGenerationResult

  // Async mode, check if there are any launchable schedule results
  if (enable_async_) {
    if (planning_sched_results_[multi_batch_id] != nullptr) {
      std::shared_ptr<ScheduleResult> result = planning_sched_results_[multi_batch_id];
      std::vector<std::shared_ptr<InferRequest>> stopped_reqs;
      bool launched =
          batch_scheduler_->TryToLaunchPlannedScheduleOutput(multi_batch_id, *(result->schedule_output), stopped_reqs);
      // launch 被拒绝时保留 planning_sched_results_, 等下一次 generation 回流后重试.
      // 典型场景: pipeline 已满, 或 fast path 下某 req 仍有 inflight 暂不可 launch.
      if (!launched) {
        return;
      }
      if (!result->schedule_output->running_reqs.empty()) {
        KLLM_LOG_SCHEDULER << "Put result running_reqs size=" << result->schedule_output->running_reqs.size();
        ProcessScheduleDataInternal(multi_batch_id, *result);
        sched_result_queue_.Put(result);
      } else {
        KLLM_LOG_SCHEDULER << "Drop result because running_reqs is empty";
      }
      for (const auto& req : stopped_reqs) {
        RemoveRequestFromQueue(result->generation_output_group.reqs[req->attn_dp_group_id], req);
      }
      planning_sched_results_[multi_batch_id] = nullptr;
      // Notify all waiting AsyncScheduleThread that the planning result has been consumed
      schedule_guard_cv_.notify_all();
    }
  } else {
    KLLM_CHECK(!sync_generating_finished_[multi_batch_id]);
    sync_generating_finished_[multi_batch_id] = true;
    // Notify all waiting SyncScheduleThread that the generation result has been consumed
    schedule_guard_cv_.notify_all();
  }
}

void ScheduleProcessor::Lock() {
  if (batch_scheduler_ != nullptr) {
    batch_scheduler_->Lock();
  }
}

void ScheduleProcessor::Unlock() {
  if (batch_scheduler_ != nullptr) {
    batch_scheduler_->Unlock();
  }
}

void ScheduleProcessor::SyncScheduleThread(size_t multi_batch_id) {
  while (!terminated_) {
    std::shared_ptr<ScheduleResult> result = GetScheduleResultInSync(multi_batch_id);
    sched_result_queue_.Put(result);

    if (terminated_) {
      return;
    }
  }
}

void ScheduleProcessor::AsyncScheduleThread(size_t multi_batch_id) {
  while (!terminated_) {
    // Check if there's existing planning result that needs to be consumed first
    {
      std::unique_lock<std::mutex> lock(schedule_guard_);
      // Wait until planning_sched_results_[multi_batch_id] is nullptr or thread is terminated
      schedule_guard_cv_.wait(
          lock, [this, multi_batch_id]() { return planning_sched_results_[multi_batch_id] == nullptr || terminated_; });
    }

    if (terminated_) {
      break;
    }

    std::shared_ptr<ScheduleResult> result = Schedule(multi_batch_id);
    if (terminated_) {
      sched_result_queue_.Put(nullptr);
      return;
    }

    std::vector<std::shared_ptr<InferRequest>> stopped_reqs;
    if (batch_scheduler_->TryToLaunchPlannedScheduleOutput(multi_batch_id, *(result->schedule_output), stopped_reqs)) {
      if (!result->schedule_output->running_reqs.empty()) {
        KLLM_LOG_SCHEDULER << "Put result running_reqs size=" << result->schedule_output->running_reqs.size();
        ProcessScheduleDataInternal(multi_batch_id, *result);
        sched_result_queue_.Put(result);
      } else {
        KLLM_LOG_SCHEDULER << "Drop result because running_reqs is empty";
      }
    } else {
      if (result->schedule_output->running_reqs.size() > 0) {
        std::lock_guard<std::mutex> lock(schedule_guard_);
        assert(planning_sched_results_[multi_batch_id] == nullptr);
        planning_sched_results_[multi_batch_id] = result;
      }
    }
    for (const auto& req : stopped_reqs) {
      RemoveRequestFromQueue(result->generation_output_group.reqs[req->attn_dp_group_id], req);
    }
  }
}

std::shared_ptr<ScheduleResult> ScheduleProcessor::GetScheduleResultInSync(size_t multi_batch_id) {
  {
    std::unique_lock<std::mutex> lock(schedule_guard_);
    // Wait until sync_generating_finished_[multi_batch_id] is true or thread is terminated
    schedule_guard_cv_.wait(
        lock, [this, multi_batch_id]() { return sync_generating_finished_[multi_batch_id] || terminated_; });
  }

  // Sync mode
  std::shared_ptr<ScheduleResult> result = Schedule(multi_batch_id);
  if (terminated_) {
    return nullptr;
  }

  std::vector<std::shared_ptr<InferRequest>> stopped_reqs;
  KLLM_CHECK(
      batch_scheduler_->TryToLaunchPlannedScheduleOutput(multi_batch_id, *(result->schedule_output), stopped_reqs));
  ProcessScheduleDataInternal(multi_batch_id, *result);
  sync_generating_finished_[multi_batch_id] = false;
  return result;
}

std::shared_ptr<ScheduleResult> ScheduleProcessor::Schedule(size_t multi_batch_id) {
  std::shared_ptr<ScheduleResult> result;
  std::shared_ptr<ScheduleOutputGroup> schedule_output_group;
  while (!terminated_) {
    // 1. Call Schedule directly
    schedule_output_group = batch_scheduler_->Schedule(multi_batch_id);

    // 2. Check if there are any running requests
    if (schedule_output_group->RunningSize() == 0) {
      // No running requests, need to wait
      if (batch_scheduler_->IsIdle(multi_batch_id)) {
        batch_scheduler_->WaitUntilHaveReqs(multi_batch_id);
      } else {
        KLLM_LOG_DEBUG << "multi_batch_id=" << multi_batch_id << " not idle, sleep 100ms";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      continue;  // Continue the loop to wait
    }
    break;
  }

  if (terminated_) {
    return nullptr;
  }

  result = std::make_shared<ScheduleResult>();
  result->outputs = schedule_output_group->outputs;

  // 3. Merge schedule results
  result->schedule_output = std::make_shared<ScheduleOutput>();
  MergeScheduleOutputGroupRunningRequests(schedule_output_group, *(result->schedule_output));

  // 4. There are running requests, process the data
  result->generation_output_group.BuildFromScheduleOutputGroup(*schedule_output_group);
  result->generation_output_group.multi_batch_id = multi_batch_id;

  //  schedule output in schedule_output_group have been used, clear running_reqs to avoid blocking schedule.
  for (auto& scheduled_out : schedule_output_group->outputs) {
    scheduled_out->ClearRunningReqs();
  }
  return result;
}

Status ScheduleProcessor::ProcessScheduleDataInternal(size_t multi_batch_id, ScheduleResult& result) {
  // TODO(robertyuan): deep copy running_reqs to avoid InferRequest changesin async scheduling
  result.running_reqs.clear();
  for (auto& req : result.schedule_output->running_reqs) {
    req->accepted_tokens.clear();
    req->generated_tokens.clear();
    result.running_reqs[req->req_id] = req;
  }
  return Status();
}

void ScheduleProcessor::Start() {}

void ScheduleProcessor::Stop() {
  terminated_ = true;

  batch_scheduler_->Stop();

  // Notify all waiting threads to wake up and check terminated_ flag
  schedule_guard_cv_.notify_all();

  // Stop async threads
  for (auto& thread : sched_threads_) {
    thread.join();
  }
  KLLM_LOG_INFO << "ScheduleProcessor stopped";
}

}  // namespace ksana_llm
