/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "executor/executor_request_manager.h"
#include "executor/inflight_resource.h"
#include "executor/model_builder.h"
#include "executor/model_executor.h"
#include "executor/model_sampler.h"
#include "runtime/generation_controller.h"
#include "runtime/hidden_unit_buffer_pool.h"
#include "runtime/structured_generation/structured_generator_factory.h"
#include "runtime/structured_generation/structured_generator_interface.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/waiter.h"

namespace ksana_llm {

// The runtime instance, used to execute the real inference process.
// It does not contain any thread itself, but invoked by other pipeline thread.
class ExecutorRuntime {
 public:
  ExecutorRuntime(std::shared_ptr<Environment> env, std::shared_ptr<Context> context);

  // Create model_builder & model_executor & model_sampler.
  Status Initialize();

  void SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime);
  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);
  void SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight);
  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager);
  void SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager);
  void SetHiddenUnitBufferPool(std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool);
  void SetDraftGeneratorController(std::shared_ptr<DraftGeneratorController> draft_generator_controller);

  // Get structured generator factory for creating generators
  std::shared_ptr<StructuredGeneratorFactory> GetStructuredGeneratorFactory() { return structured_generator_factory_; }

  std::shared_ptr<GenerationController> GetGenerationController() { return generation_controller_; }

  // 暴露 model_sampler，供 ModelRunner 在 build-ahead 同步路径（Forward→Sampling 同线程）使用。
  std::shared_ptr<ModelSampler> GetModelSampler() const { return model_sampler_; }

  // The inflight resource manager drives automatic allocation and release of per-request
  // runtime resources (e.g. compressor slots). owned by ExecutorRuntime.
  std::shared_ptr<InflightResourceManager> GetInflightResourceManager() { return inflight_resource_mgr_; }

  // pd_v2 KV dispatcher uses this to enumerate live ExecutorInferRequests
  // on each per-layer event. Returns the same shared instance held inside.
  ExecutorRequestManager* GetExecutorRequestManager() { return executor_request_manager_.get(); }


  Status BuildModelInput(std::shared_ptr<ScheduleOutput> schedule_output, std::shared_ptr<NewModelInput> model_input,
                         const RunMode run_mode);
  // build-ahead: Preprocess 线程在 slot 对应 H2D stream 上 ParseFromRequests，与 Execute forward 重叠。
  Status PrepareModelInputGpu(std::shared_ptr<NewModelInput> model_input);
  Status Forward(std::shared_ptr<NewModelInput> model_input, HiddenUnitDeviceBuffer* hidden_unit,
                 std::shared_ptr<NewModelOutput> model_output);
  Status Sampling(std::shared_ptr<NewModelOutput> model_output, std::shared_ptr<SamplingOutput> sampling_output);
  Status MtpForward(const size_t mtp_step_num, std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                    HiddenUnitDeviceBuffer* hidden_unit, std::shared_ptr<ScheduleOutput> schedule_output);
  Status DSparkForward(const size_t dspark_block_size, const int dspark_noise_token_id,
                       std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs, HiddenUnitDeviceBuffer* hidden_unit,
                       std::shared_ptr<ScheduleOutput> schedule_output);

  // Cache full requests, and update incremental requests.
  void UpdateCachedRequests(std::shared_ptr<ScheduleOutput> schedule_output);

  // Apply incremental patch, get full schedule_output again.
  void RebuildScheduleOutput(std::shared_ptr<ScheduleOutput> schedule_output);

  // Wake any thread blocked inside Forward()'s prev_mtp_done_->Wait(). Must be called by the owner
  // (ModelRunner::Stop) after the input/output queues have been stopped, so that the execute thread
  // is not stuck waiting on an MTP Notify that will never come during shutdown.
  void Shutdown();

 private:
  Status InitializeStructuredGeneratorFactory();

  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;

  std::shared_ptr<ModelBuilder> model_builder_ = nullptr;
  std::shared_ptr<ModelExecutor> model_executor_ = nullptr;
  std::shared_ptr<ModelSampler> model_sampler_ = nullptr;

  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> model_weight_ = nullptr;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_ = nullptr;
  // The SWA block allocator manager; nullptr when SWA is disabled.
  std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager_ = nullptr;

  // The hidden buffer used for manager.
  std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool_ = nullptr;

  // The inflight resource manager.
  std::shared_ptr<InflightResourceManager> inflight_resource_mgr_ = nullptr;

  // Cache all necessary request informations.
  std::shared_ptr<ExecutorRequestManager> executor_request_manager_ = nullptr;

  // Structured generator factory for creating generators on Executor side
  std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory_ = nullptr;

  // The generation controller for inference engine.
  std::shared_ptr<GenerationController> generation_controller_ = nullptr;

  // Prevent concurrent modification of infer_request by MTPForward and UpdateInferRequest
  std::mutex infer_req_mutex_;

  // True when this runtime instance must serialize the main forward (HandleExecute thread,
  // RunMode::kMain) against the immediately preceding MTP forward (HandlePostprocess thread,
  // RunMode::kNextN) on the same rank. Required to avoid the cross-TP-rank deadlock in
  // CommonModel::forward_mutex_ when TP > 1 and the same TrtAllReduce/AllReduce kernel is launched
  // for two different RunModes in inconsistent order on different TP ranks. Computed once at
  // construction from (IsSamplingNode && mtp_step_num > 0 && max_pp_batch_num > 1).
  const bool serialize_main_mtp_{false};

  // Used only when serialize_main_mtp_ is true: HandleExecute path (Forward) waits on this before
  // launching the next main forward; HandlePostprocess path (MtpForward) notifies it at the end.
  // Initialized to a "ready" state so the first main Forward proceeds immediately. Internally
  // backed by std::mutex + std::condition_variable; we use Waiter to keep that out of this header.
  std::shared_ptr<Waiter> prev_mtp_done_;
};

}  // namespace ksana_llm
