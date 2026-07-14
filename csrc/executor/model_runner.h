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
#include "runtime/weight_instance_inferface.h"
#include "utils/blocking_queue.h"

namespace ksana_llm {

class ExecutorBlockManager;

// Used to execute really inference calculation on the device.
class ModelRunner {
 public:
  explicit ModelRunner(std::shared_ptr<Context> context);

  void SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper);
  void SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime);
  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);
  void SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight);
  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager);
  void SetExecutorBlockManager(std::shared_ptr<ExecutorBlockManager> executor_block_manager);
  void SetHiddenUnitBufferPool(std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool);
  void SetSamplingOutputPool(std::shared_ptr<SamplingOutputPool> sampling_output_pool);
  void SetDraftGeneratorController(std::shared_ptr<DraftGeneratorController> draft_generator_controller);

  // Start runner, all worker threads should be ready.
  Status Start();

  // Stop runner, all thread must exit gracefully.
  Status Stop();

 private:
  Status HandlePreprocess();
  Status HandleExecute();
  Status HandlePostprocess();
  Status HandleResultReportProcess();
  Status HandleDistributedScheduleOutput();

  void GetHiddenShape(const std::shared_ptr<ScheduleOutput>& schedule_output, std::vector<size_t>& shape);
  DataType GetHiddenDataType();

 private:
  std::shared_ptr<Context> context_ = nullptr;

  // The ipc func wrapper, used to invoke functions between processes.
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_ = nullptr;

  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> model_weight_ = nullptr;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_ = nullptr;
  std::shared_ptr<ExecutorBlockManager> executor_block_manager_ = nullptr;

  // The really executor runtime.
  std::shared_ptr<ExecutorRuntime> executor_runtime_ = nullptr;

  std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool_ = nullptr;

  const size_t mtp_step_num_;

 private:
  // Read schedule_output and build model input.
  std::thread preprocess_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<ScheduleOutput>>> schedule_output_queue_ = nullptr;

  // Read model input and execute model forward.
  std::thread execute_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<NewModelInput>>> model_input_queue_ = nullptr;

  // Read forward output and sampling.
  std::thread postprocess_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<NewModelOutput>>> model_output_queue_ = nullptr;

  // Read sampling output and report to engine process.
  std::thread result_report_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<SamplingOutput>>> sampling_output_queue_ = nullptr;

  std::shared_ptr<SamplingOutputPool> sampling_output_pool_ = nullptr;

  std::atomic<bool> terminated_ = false;
};

}  // namespace ksana_llm
