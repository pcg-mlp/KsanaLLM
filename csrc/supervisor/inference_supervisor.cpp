/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "supervisor/inference_supervisor.h"
#include <unistd.h>

#include <memory>

#include "configure/schedule_config_parser.h"
#include "device/device_utils.h"
#include "ipc/core/ipc_process_watchdog.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"
#include "utils/string_utils.h"

#define PROCESS_LOCK_FILE "/tmp/ksana_llm.lock"

namespace ksana_llm {

InferenceSupervisor::InferenceSupervisor(const std::string &config_file, bool is_engine, int executor_id,
                                         const std::string &json_model_override_args) {
  config_file_ = config_file;
  is_engine_ = is_engine;
  executor_id_ = executor_id;

  // TODO(yancyliu): Use std::make_shared later, should remove singleton.
  env_ = Singleton<Environment>::GetInstance();
  if (!is_engine_) {
    env_->InitializeAttentionBackend();
  }

  STATUS_CHECK_FAILURE(env_->ParseConfig(config_file_, "", "config.json", json_model_override_args));

  RuntimeConfig runtime_config;
  STATUS_CHECK_FAILURE(env_->GetRuntimeConfig(runtime_config));

  // get batch schedule config;
  BatchSchedulerConfig batch_scheduler_config;
  STATUS_CHECK_FAILURE(env_->GetBatchSchedulerConfig(batch_scheduler_config));

  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
  bool is_chief = pipeline_config.world_size == 1 || pipeline_config.node_rank == 0;
  if (!is_chief) {
    batch_scheduler_config.max_pp_batch_num = 1;
    env_->SetBatchSchedulerConfig(batch_scheduler_config);
  }

  size_t tp_size = runtime_config.parallel_basic_config.tensor_parallel_size;
  size_t adp_size = runtime_config.parallel_basic_config.attn_data_parallel_size;

  // Environment must be initialized before context.
  KLLM_LOG_INFO << "Get tensor parallel: " << tp_size << " attention data parallel: " << adp_size
                << " max_pp_batch_num: " << batch_scheduler_config.max_pp_batch_num;
  context_ = std::make_shared<Context>();
  context_->global.reset(new GlobalContext(tp_size, adp_size, batch_scheduler_config.max_pp_batch_num, env_));
}

InferenceSupervisor::~InferenceSupervisor() { Destroy(); }

Status InferenceSupervisor::InitIpc() {
  // Create ipc func wrapper.
  ipc_func_wrapper_ =
      std::make_shared<IpcFuncWrapper>(context_->global->GetTensorParallelSize(), is_engine_, executor_id_);
  return Status();
}

Status InferenceSupervisor::InitService(const EndpointConfig &endpoint_config, int think_end_token_id) {
  endpoint_config_ = endpoint_config;
  context_->global->SetEngineProcess(true);

  // Set think_end_token_id in Environment if provided
  if (think_end_token_id >= 0) {
    KLLM_LOG_INFO << "InferenceSupervisor: Setting think_end_token_id to " << think_end_token_id;
    env_->SetThinkEndTokenId(think_end_token_id);
  }

  MonitorChildren();

  // Wait until all ipc executors are ready.
  std::vector<int64_t> executor_pids;
  ipc_func_wrapper_->WaitAllExecutorsReady(executor_pids);
  KLLM_LOG_INFO << "Get all executor pids: " << Vector2Str(executor_pids);
  SetMonitoredChildPids(executor_pids);

  // Start engine service.
  inference_service_ = std::make_shared<InferenceService>(endpoint_config_, env_, context_);
  inference_service_->SetIpcFuncWrapper(ipc_func_wrapper_);
  inference_service_->Initialize();

  return inference_service_->Start();
}

Status InferenceSupervisor::InitExecutor() {
  context_->global->SetEngineProcess(false);

  MonitorParent();

  // Start executor.
  inference_executor_ = std::make_shared<InferenceExecutor>(executor_id_, env_, context_);
  inference_executor_->SetIpcFuncWrapper(ipc_func_wrapper_);
  inference_executor_->Initialize();
  inference_executor_->Start();

  // Mark current executor as ready.
  ipc_func_wrapper_->AddExecutor();

  // Never return.
  inference_executor_->BlockingForever();
  return Status();
}

Status InferenceSupervisor::Destroy() {
  if (inference_service_) {
    inference_service_->Stop();
    inference_service_ = nullptr;
  }

  return Status();
}

std::shared_ptr<InferenceService> &InferenceSupervisor::GetService() {
  // Expose inference service to serving op.
  return inference_service_;
}

}  // namespace ksana_llm
