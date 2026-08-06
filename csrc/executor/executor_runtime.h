/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

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

namespace ksana_llm {

class CompositeSpeculativeDecoder;

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

  void SetCompositeSpeculativeDecoder(std::shared_ptr<CompositeSpeculativeDecoder> composite_decoder);

  // Get structured generator factory for creating generators
  std::shared_ptr<StructuredGeneratorFactory> GetStructuredGeneratorFactory() { return structured_generator_factory_; }

  std::shared_ptr<GenerationController> GetGenerationController() { return generation_controller_; }

  // 暴露 model_sampler，供 ModelRunner 在 build-ahead 同步路径（Forward→Sampling 同线程）使用。
  std::shared_ptr<ModelSampler> GetModelSampler() const { return model_sampler_; }

  // Expose sub-components for stub adapter construction during init.
  std::shared_ptr<ModelExecutor> GetModelExecutor() const { return model_executor_; }
  std::shared_ptr<ModelBuilder> GetModelBuilder() const { return model_builder_; }

  // The inflight resource manager drives automatic allocation and release of per-request
  // runtime resources (e.g. compressor slots). owned by ExecutorRuntime.
  std::shared_ptr<InflightResourceManager> GetInflightResourceManager() { return inflight_resource_mgr_; }

  // pd_v2 KV dispatcher uses this to enumerate live ExecutorInferRequests
  // on each per-layer event. Returns the same shared instance held inside.
  ExecutorRequestManager* GetExecutorRequestManager() { return executor_request_manager_.get(); }

  Status BuildModelInput(std::shared_ptr<ScheduleOutput> schedule_output, std::shared_ptr<NewModelInput> model_input);
  // build-ahead: Preprocess 线程在 slot 对应 H2D stream 上 ParseFromRequests，与 Execute forward 重叠。
  Status PrepareModelInputGpu(std::shared_ptr<NewModelInput> model_input);
  Status Forward(std::shared_ptr<NewModelInput> model_input, HiddenUnitDeviceBuffer* hidden_unit,
                 std::shared_ptr<NewModelOutput> model_output);
  Status Sampling(std::shared_ptr<NewModelOutput> model_output, std::shared_ptr<SamplingOutput> sampling_output);

  // Cache full requests, and update incremental requests.
  void UpdateCachedRequests(std::shared_ptr<ScheduleOutput> schedule_output);

  // Apply incremental patch, get full schedule_output again.
  void RebuildScheduleOutput(std::shared_ptr<ScheduleOutput> schedule_output);

  // Wake any thread blocked inside Forward()'s prev_mtp_done_->Wait(). Must be called by the owner
  // (ModelRunner::Stop) after the input/output queues have been stopped, so that the execute thread
  // is not stuck waiting on an MTP Notify that will never come during shutdown.
  void Shutdown();

  // Register a callback to be invoked at the end of Initialize(), after all
  // internal components (model_builder/executor/sampler) are fully constructed.
  // Callbacks run synchronously in registration order. Each callback runs at most once.
  void RegisterPostInitCallback(std::function<void()> cb) { post_init_cbs_.push_back(std::move(cb)); }

  // Expose infer_req_mutex_ so that MTP/DSpark decoders (which run on the Execute/Postprocess
  // thread and mutate forwarding_tokens/kv_cached_token_num/prefix_cache_len across their
  // multi-step forward loop) can synchronize against UpdateCachedRequests (Preprocess thread,
  // via UpdateInferRequest) on the same ExecutorInferRequest instances. Decoders must lock this
  // exact mutex instance rather than a private one, or the two threads race on those fields.
  std::mutex& GetInferReqMutex() { return infer_req_mutex_; }

 private:
  Status InitializeStructuredGeneratorFactory();

  // Post-init callbacks: invoked synchronously at end of Initialize().
  std::vector<std::function<void()>> post_init_cbs_;

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

  std::shared_ptr<CompositeSpeculativeDecoder> composite_decoder_ = nullptr;

  // Prevent concurrent modification of infer_request by spec decoders' ExecuteForward (MTP/DSpark)
  // and UpdateInferRequest (Preprocess thread).
  std::mutex infer_req_mutex_;
};

}  // namespace ksana_llm
