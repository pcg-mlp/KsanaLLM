/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "configure/global_context.h"

#include <iostream>
#include <stdexcept>

#include "configure/environment.h"
#include "device/device_types.h"
#include "device/device_utils.h"
#include "fmt/core.h"
#include "utils/logger.h"
#include "utils/singleton.h"

namespace ksana_llm {

GlobalContext::GlobalContext(const size_t tensor_parallel_size, const size_t attn_data_parallel_size,
                             const size_t max_multi_batch_size, std::shared_ptr<Environment> env)
    : tensor_parallel_size_(tensor_parallel_size), attn_data_parallel_size_(attn_data_parallel_size) {
  env_ = env;
}

void GlobalContext::InitDistributedConfig() {
  // Initialize pipeline configure.
  PipelineConfig pipeline_config_;
  env_->GetPipelineConfig(pipeline_config_);

  world_size_ = pipeline_config_.world_size;
  node_rank_ = pipeline_config_.node_rank;

  master_host_ = pipeline_config_.master_host;
  master_port_ = pipeline_config_.master_port;

  is_standalone_ = (world_size_ == 1);
  is_distributed_master_ = (world_size_ > 1) && (node_rank_ == 0);
  is_distributed_worker_ = (world_size_ > 1) && (node_rank_ > 0);
  is_distributed_last_worker_ = (world_size_ > 1) && (node_rank_ == (world_size_ - 1));

  // Initialize expert parallel configure.  Make sure ep share is_chief_ with pp is ok later.
  env_->GetExpertParallelConfig(expert_parallel_config_);
  is_expert_chief_ = expert_parallel_config_.expert_world_size == 1 || expert_parallel_config_.expert_node_rank == 0;
  is_expert_standalone_ = expert_parallel_config_.expert_world_size == 1;
}

size_t GlobalContext::GetMtpStepNum() const {
  BatchSchedulerConfig config;
  env_->GetBatchSchedulerConfig(config);
  return config.mtp_step_num;
}

size_t GlobalContext::GetMaxPpBatchNum() const {
  BatchSchedulerConfig config;
  env_->GetBatchSchedulerConfig(config);
  return config.max_pp_batch_num;
}

void GlobalContext::SetEngineProcess(bool is_engine) { is_engine_ = is_engine; }

bool GlobalContext::IsEngine() const { return is_engine_; }

bool GlobalContext::IsStandalone() const { return is_standalone_; }

bool GlobalContext::IsDistributedMaster() const { return is_distributed_master_; }

bool GlobalContext::IsDistributedWorker() const { return is_distributed_worker_; }

bool GlobalContext::IsDistributedLastWorker() const { return is_distributed_last_worker_; }

bool GlobalContext::IsDistributedResultReporter() const { return is_distributed_last_worker_; }

bool GlobalContext::IsExpertParallelStandalone() const { return is_expert_standalone_; }

bool GlobalContext::IsExpertParallelChief() const { return is_expert_chief_; }

bool GlobalContext::IsSamplingNode() const { return is_standalone_ || is_distributed_last_worker_; }

}  // namespace ksana_llm
