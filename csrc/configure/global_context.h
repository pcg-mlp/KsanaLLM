/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>

#include "configure/environment.h"
#include "configure/schedule_config_parser.h"

namespace ksana_llm {

class GlobalContext {
 public:
  GlobalContext(const size_t tensor_parallel_size, const size_t attn_data_parallel_size,
                const size_t max_multi_batch_size, std::shared_ptr<Environment> env);

  size_t GetTensorParallelSize() { return tensor_parallel_size_; }
  size_t GetAttentionTensorParallelSize() { return tensor_parallel_size_ / attn_data_parallel_size_; }
  size_t GetAttentionDataParallelSize() { return attn_data_parallel_size_; }

  inline bool IsRunContextDecodeAndDecodeSerially() { return is_contextdecode_and_decode_run_serially_; }

  inline bool IsGemmFp8Supported() { return is_gemm_fp8_supported_; }

  // Whether current node is standalone mode.
  bool IsStandalone() const;

  // Whether current node is distributed master.
  bool IsDistributedMaster() const;

  // Whether current node is distributed worker.
  bool IsDistributedWorker() const;

  // Whether current node is distributed last worker.
  bool IsDistributedLastWorker() const;

  // Whether current node is result reporter.
  bool IsDistributedResultReporter() const;

  // Whether current node is standalone mode in Expert-Parallel.
  bool IsExpertParallelStandalone() const;

  // Whether current node is master node in Expert-Parallel.
  bool IsExpertParallelChief() const;

  // Whether current node need sampling.
  bool IsSamplingNode() const;

  // Total number of machines participating in Expert-Parallel
  size_t GetExpertParallelWorldSize() { return expert_parallel_config_.expert_world_size; }

  // Rank value of the current node in multi-machine Expert-Parallel scenario
  size_t GetExpertParallelExpertNodeRank() { return expert_parallel_config_.expert_node_rank; }

  // Reset chielf and standalone flag.
  void InitDistributedConfig();

  void SetEngineProcess(bool is_engine);
  bool IsEngine() const;

  size_t GetWorldSize() { return world_size_; }
  size_t GetNodeRank() { return node_rank_; }

  std::string GetMasterHost() { return master_host_; }
  uint16_t GetMasterPort() { return master_port_; }

  size_t GetMtpStepNum() const;

 private:
  // The environments.
  std::shared_ptr<Environment> env_ = nullptr;

  const size_t tensor_parallel_size_{0};
  const size_t attn_data_parallel_size_{0};
  static constexpr int defalt_device_id_{0};

  // if true, only one thread execute context_decode/decode and context_decode decode run in sync
  bool is_contextdecode_and_decode_run_serially_{true};
  bool is_gemm_fp8_supported_{false};

  // pipeline config.
  ExpertParallelConfig expert_parallel_config_;

  bool is_expert_standalone_ = false;

  bool is_standalone_ = false;
  bool is_distributed_master_ = false;
  bool is_distributed_worker_ = false;
  bool is_distributed_last_worker_ = false;

  // Whether current process is engine process.
  bool is_engine_ = false;

  // The world size and node rank.
  size_t world_size_;
  size_t node_rank_;

  // The master host and port.
  std::string master_host_;
  uint16_t master_port_;

  // Single node or master node of expert parallel mode.
  bool is_expert_chief_ = false;
};

}  // namespace ksana_llm
