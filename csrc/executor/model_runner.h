/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "device/device_context.h"
#include "executor/executor_runtime.h"
#include "ipc/ipc_func_wrapper.h"
#include "runtime/model_input.h"
#include "runtime/model_instance.h"
#include "runtime/model_output.h"
#include "runtime/sampling_output.h"
#include "runtime/schedule_output.h"
#include "runtime/speculative/orchestration/composite_speculative_decoder.h"
#include "runtime/weight_instance_inferface.h"
#include "utils/blocking_queue.h"

namespace ksana_llm {

class GlobalCachePutWorker;

// Used to execute really inference calculation on the device.
class ModelRunner {
 public:
  explicit ModelRunner(std::shared_ptr<Context> context);

  void SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper);
  void SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime);
  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);
  void SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight);
  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager);
  void SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager);
  void SetHiddenUnitBufferPool(std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool);
  void SetSamplingOutputPool(std::shared_ptr<SamplingOutputPool> sampling_output_pool);
  void SetCompositeSpeculativeDecoder(std::shared_ptr<CompositeSpeculativeDecoder> decoder);
  void SetGlobalCachePutWorker(std::shared_ptr<GlobalCachePutWorker> global_cache_put_worker);

  // Start runner, all worker threads should be ready.
  Status Start();

  // Stop runner, all thread must exit gracefully.
  Status Stop();

 private:
  Status HandlePreprocess();
  Status HandleExecute();
  // 慢路径（build-ahead 关闭）: Execute 只跑 Forward，Postprocess 独立线程做 Sampling。
  Status HandlePostprocess();
  // build-ahead 路径: Execute 内单次调用 HandlePostprocess（不经 model_output_queue_）。
  Status HandlePostprocessInline(std::shared_ptr<NewModelOutput> model_output);
  Status HandleResultReportProcess();
  Status HandleDistributedScheduleOutput();

  void GetHiddenShape(const std::shared_ptr<ScheduleOutput>& schedule_output, std::vector<size_t>& shape);
  DataType GetHiddenDataType();

  // build-ahead + TP: rank 0 WriteRing 后，所有 TP rank 集体广播 ring slot。
  void BroadcastBuildAheadRingSlot(size_t slot);

  // build-ahead: Sampling 后将本步 token 写入 ring slot（仅 postprocess_enqueue_report_ 为 true 时执行）。
  void WriteBuildAheadRingAfterSampling(const std::shared_ptr<NewModelOutput>& model_output);

 private:
  std::shared_ptr<Context> context_ = nullptr;

  // The ipc func wrapper, used to invoke functions between processes.
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_ = nullptr;

  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> model_weight_ = nullptr;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_ = nullptr;

  // The really executor runtime.
  std::shared_ptr<ExecutorRuntime> executor_runtime_ = nullptr;

  // Raw pointer to the InflightResourceManager owned by executor_runtime_.
  // Cached here to avoid a shared_ptr copy (and its atomic refcount operations)
  // on the hot preprocess path. Lifetime: executor_runtime_ outlives all
  // worker threads, so this pointer is valid throughout HandlePreprocess.
  InflightResourceManager* inflight_resource_mgr_ = nullptr;

  std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool_ = nullptr;

 private:
  // Read schedule_output and build model input.
  std::thread preprocess_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<ScheduleOutput>>> schedule_output_queue_ = nullptr;

  std::thread execute_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<NewModelInput>>> model_input_queue_ = nullptr;

  // 慢路径: Execute → model_output_queue_ → HandlePostprocess。
  std::thread postprocess_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<NewModelOutput>>> model_output_queue_ = nullptr;

  // build-ahead: HandlePostprocessInline → sampling_output_queue_ → HandleResultReportProcess。
  std::thread result_report_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<SamplingOutput>>> sampling_output_queue_ = nullptr;

  // build-ahead 单次 postprocess 的跨调用状态（仅 Execute 线程经 HandlePostprocessInline 写入）。
  // 慢路径 postprocess 线程从不触碰这些字段，与 model_output_queue_ 消费路径互斥。
  bool postprocess_one_shot_ = false;
  // true：本步结果入 sampling_output_queue_，由 HandleResultReportProcess 异步上报（build-ahead）；
  // false：postprocess 线程在 device 0 上同步 IPC 上报（慢路径三线程模型）。
  bool postprocess_enqueue_report_ = false;
  std::shared_ptr<NewModelOutput> postprocess_one_shot_output_ = nullptr;

  std::shared_ptr<SamplingOutputPool> sampling_output_pool_ = nullptr;

  std::atomic<bool> terminated_ = false;

  // 全局 cache put 工作器（Forward 完成后异步上传 KV cache）
  std::shared_ptr<GlobalCachePutWorker> global_cache_put_worker_ = nullptr;

  // 本步采样 ring 的当前 slot；仅在 HandlePreprocess 线程内访问，无需原子。
  // build-ahead 启用时每步 cur_ring_slot_ ^= 1u（0/1 交替），上一步写入 slot 为 cur_slot ^ 1u；
  // 不依赖 batch 是否同构或本步是否做替换（req_id 解耦，prefill/decode 混合一视同仁）。
  size_t cur_ring_slot_ = 0;

  // Composite speculative decoder (unified interface for all speculative methods).
  std::shared_ptr<CompositeSpeculativeDecoder> composite_decoder_ = nullptr;
};

}  // namespace ksana_llm
