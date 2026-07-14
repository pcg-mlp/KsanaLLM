/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>
#include "device/device_utils.h"
#include "distributed/control_message.h"
#include "utils/logger.h"

// 只在 DeepEP 可用时包含相关头文件
#ifdef KSANA_WITH_DEEPEP
#  include "distributed/expert_parallel/deepep/deep_ep.h"
#endif

namespace ksana_llm {

// Forward declaration
class Tensor;

class DeepEPWrapper {
 public:
  // Constructor: initialize DeepEP buffer without sync
  DeepEPWrapper(int rank, int num_ranks, int world_size, int node_rank, int hidden_size, int num_experts);

  ~DeepEPWrapper();

  // Get communicator config from buffer
  EPCommunicationConfig GetCommunicationConfig();

  // Initialize with aggregated communicator config from all ranks
  void Initialize(const std::vector<EPCommunicationConfig>& comm_configs);

  void Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors, Stream stream);

  void Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors, Stream stream);

 private:
  void InitConfig();

  void SetNumSMs(int new_num_sms);

#ifdef KSANA_WITH_DEEPEP
  void DispatchLayout(torch::Tensor& topk_idx);

  void IntraNodeDispatch(const torch::Tensor& hidden_states, const std::optional<torch::Tensor>& hidden_states_scales,
                         torch::Tensor& topk_idx, torch::Tensor& topk_weights);

  void InterNodeDispatch(const torch::Tensor& hidden_states, const std::optional<torch::Tensor>& hidden_states_scales,
                         torch::Tensor& topk_idx, torch::Tensor& topk_weights);

  void IntraNodeCombine();

  void InterNodeCombine();
#endif

 private:
  constexpr static int kNumMaxNvlPeers = 8;

  // Rank information
  int local_rank_;
  int rank_;
  int num_ranks_;
  int deepep_rank_;
  int world_size_;
  int node_rank_;

  // Configuration
  int num_sms_;
  int hidden_size_;
  int num_experts_;

#ifdef KSANA_WITH_DEEPEP
  // DeepEP buffer and config
  std::shared_ptr<expert_parallel::deepep::Buffer> buffer_ = nullptr;
  std::shared_ptr<deep_ep::Config> dispatch_config_ = nullptr;
  std::shared_ptr<deep_ep::Config> combine_config_ = nullptr;
  std::optional<deep_ep::EventHandle> empty_event_ = std::nullopt;

  // Input tensors (set by calling code before Dispatch/Combine)
  torch::Tensor hidden_states_;
  std::optional<torch::Tensor> hidden_states_scales_;
  torch::Tensor topk_ids_;
  torch::Tensor topk_weights_;
  void* output_ = nullptr;

  // Dispatch layout tensors
  torch::Tensor num_tokens_per_rank_;
  torch::Tensor num_tokens_per_rdma_rank_;
  torch::Tensor num_tokens_per_expert_;
  torch::Tensor is_token_in_rank_;

  // Dispatch & combine result
  int recv_num_tokens_;

  // Intranode dispatch tensors
  torch::Tensor send_head_;
  torch::Tensor rank_prefix_matrix_;
  torch::Tensor channel_prefix_matrix_;
  torch::Tensor src_idx_;

  // Internode dispatch tensors
  torch::Tensor rdma_channel_prefix_matrix_;
  torch::Tensor gbl_channel_prefix_matrix_;
  torch::Tensor recv_gbl_rank_prefix_sum_;
  torch::Tensor recv_rdma_channel_prefix_matrix_;
  torch::Tensor recv_rdma_rank_prefix_sum_;
  torch::Tensor recv_gbl_channel_prefix_matrix_;
  torch::Tensor recv_src_meta_;
  torch::Tensor send_rdma_head_;
  torch::Tensor send_nvl_head_;

 public:
  int GetRecvNumTokens() const { return recv_num_tokens_; }

  const torch::Tensor& GetRecvTopkIds() const { return topk_ids_; }

  const torch::Tensor& GetRecvTopkWeights() const { return topk_weights_; }

  const std::optional<torch::Tensor>& GetRecvHiddenStatesScales() const { return hidden_states_scales_; }
#else

 public:
  int GetRecvNumTokens() const { KLLM_THROW("DeepEP is not available. GetRecvNumTokens() cannot be called."); }

  const torch::Tensor& GetRecvTopkIds() const {
    KLLM_THROW("DeepEP is not available. GetRecvTopkIds() cannot be called.");
  }

  const torch::Tensor& GetRecvTopkWeights() const {
    KLLM_THROW("DeepEP is not available. GetRecvTopkWeights() cannot be called.");
  }

  const std::optional<torch::Tensor>& GetRecvHiddenStatesScales() const {
    KLLM_THROW("DeepEP is not available. GetRecvHiddenStatesScales() cannot be called.");
  }
#endif
};

}  // namespace ksana_llm
