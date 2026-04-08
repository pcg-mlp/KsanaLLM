/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <thread>

#include "batch_scheduler/schedule_processor_interface.h"
#include "distributed/distributed_control_coordinator.h"
#include "ipc/ipc_func_wrapper.h"
#include "runtime/sampling_output.h"
#include "runtime/schedule_output.h"
#include "transfer/transfer_engine.h"
#include "utils/blocking_queue.h"
#include "utils/channel.h"
#include "utils/request.h"
#include "utils/status.h"

namespace ksana_llm {

// Used to manage all engine threads and queues.
class ModelDriver {
 public:
  ModelDriver(Channel<std::pair<Status, std::shared_ptr<Request>>>& request_queue,
              std::shared_ptr<ScheduleOutputPool> schedule_output_pool,
              std::shared_ptr<SamplingOutputPool> sampling_output_pool, std::shared_ptr<Context> context);

  void SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper);

  // Set transfer engine for token transmission
  void SetTransferEngine(std::shared_ptr<TransferEngine> transfer_engine);

  // Initialize communication between multiple devices such as nccl.
  Status InitDeviceCommunication();
  Status GetDeviceMemory(size_t* device_free, size_t* device_total);

  // Initialize custom communication.
  Status InitCustomCommunication();

  // Initialize multicast communication.
  Status InitMulticastCommunication();

  // Load weight and model, blocked until all executor finished.
  Status LoadWeight();
  Status LoadModel();

  // Reset pipeline config on executor process.
  Status ResetPipelineConfig(const PipelineConfig& pipeline_config);

  // TODO(robertyuan): temp function, remove later
  void SetModelConfigAndRuntimeConfig(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
    model_config_ = model_config;
    runtime_config_ = runtime_config;
  }

  // Build data communications between executors.
  Status ConnectDataChannels(std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator);

  // Synchronize EP communicator config across all nodes.
  // Get config from executors, synchronize across nodes, and set back to executors.
  Status SynchronizeEPCommunicationConfig(
      std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator);

  // Create control channel used for result report.
  Status InitializeResultReporter(std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator);

  // Create block manager on executor process.
  Status CreateBlockManager(const BlockAllocatorManagerConfig& block_allocator_manager_config);

  // Create executor runtime on executor process.
  Status CreateExecutorRuntime(const VocabInfo& vocab_info);

  // Start driver, all worker threads should be ready.
  Status Start();

  // Stop driver, all thread must exit gracefully.
  Status Stop();

  void SetScheduleProcessor(std::shared_ptr<ScheduleProcessorInterface> schedule_processor);

  // PD 分离：设置 GlobalCache 角色（"prefill" / "decode" / ""）
  void SetGlobalCacheRole(const std::string& role);

 private:
  Status Enqueue(std::shared_ptr<Request>& req);
  void UpdateSamplingOutput(std::shared_ptr<SamplingOutput> sampling_output);
  Status HandleRequestRecv();
  Status HandleSchedule();
  Status HandleScheduleOutputSend();
  Status HandleScheduleUpdate();

  // For distributed worker only.
  Status HandleDistributedScheduleOutput();

  // For distributed master only.
  Status HandleDistributedSamplingOutput();

  void SetInferRequestTransferMode(std::shared_ptr<ScheduleOutput> schedule_output);

  // Convert multimodal embedding data from py::object to torch::Tensor
  void ConvertMultimodalEmbedding(std::shared_ptr<Request>& req);

 private:
  // The channel used to pass request from endpoint.
  Channel<std::pair<Status, std::shared_ptr<Request>>>& request_queue_;
  std::shared_ptr<Context> context_ = nullptr;

  // The ipc func wrapper, used to invoke functions between processes.
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_ = nullptr;

  // Transfer engine for token transmission between prefill and decode nodes
  std::shared_ptr<TransferEngine> transfer_engine_ = nullptr;

  std::shared_ptr<ScheduleProcessorInterface> schedule_processor_ = nullptr;

  // <schedule_id, ScheduleResult>
  std::unordered_map<size_t, std::shared_ptr<ScheduleResult>> schedule_result_map_;
  std::mutex schedule_result_map_mutex_;

  // object pool of schedule and sampling output buffer.
  std::shared_ptr<ScheduleOutputPool> schedule_output_pool_ = nullptr;
  std::shared_ptr<SamplingOutputPool> sampling_output_pool_ = nullptr;

  // Recv incoming requests to schedule queue.
  std::unique_ptr<std::thread> request_recv_thread_ = nullptr;

  // The Schedule() function should be blocking.
  std::unique_ptr<std::thread> schedule_thread_ = nullptr;

  // Recv schedule output from master and send to executor.
  std::unique_ptr<std::thread> distribted_worker_thread_ = nullptr;

  // Recv sampling output from last worker and update schedule state.
  std::unique_ptr<std::thread> distribted_master_thread_ = nullptr;

  std::unique_ptr<std::thread> schedule_output_send_thread_ = nullptr;
  std::shared_ptr<BlockingQueue<std::shared_ptr<ScheduleOutput>>> schedule_output_queue_ = nullptr;

  std::unique_ptr<std::thread> schedule_update_thread_ = nullptr;
  std::shared_ptr<BlockingQueue<std::shared_ptr<SamplingOutput>>> sampling_output_queue_ = nullptr;

  std::atomic<bool> terminated_ = false;

  ModelConfig model_config_;
  RuntimeConfig runtime_config_;

  // PD 分离：GlobalCache 角色，"prefill" 时在调度完成后发送 prefill 通知
  std::string global_cache_role_;
};

}  // namespace ksana_llm
