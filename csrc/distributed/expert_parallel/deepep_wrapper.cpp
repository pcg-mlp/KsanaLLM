/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "distributed/expert_parallel/deepep_wrapper.h"

#include "utils/logger.h"
#include "utils/tensor.h"

namespace ksana_llm {

#ifdef KSANA_WITH_DEEPEP
static torch::Tensor TransToTorchTensor(const Tensor& tensor) {
  std::vector<int64_t> sizes;
  for (size_t dim : tensor.shape) {
    sizes.push_back(static_cast<int64_t>(dim));
  }

  torch::ScalarType dtype;
  switch (tensor.dtype) {
    case TYPE_FP16:
      dtype = torch::kFloat16;
      break;
    case TYPE_FP32:
      dtype = torch::kFloat32;
      break;
    case TYPE_FP8_E4M3:
      dtype = torch::kFloat8_e4m3fn;
      break;
    case TYPE_INT32:
      dtype = torch::kInt32;
      break;
    case TYPE_INT64:
      dtype = torch::kInt64;
      break;
    case TYPE_BF16:
      dtype = torch::kBFloat16;
      break;
    default:
      KLLM_LOG_ERROR << "Unsupported tensor dtype: " << tensor.dtype;
      dtype = torch::kFloat16;
  }

  // data has already in GPU, directly from_blob
  return torch::from_blob(tensor.GetPtr<void>(), sizes, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
}
#endif  // KSANA_WITH_DEEPEP

DeepEPWrapper::DeepEPWrapper(int rank, int num_ranks, int world_size, int node_rank, int hidden_size, int num_experts) {
#ifdef KSANA_WITH_DEEPEP
  local_rank_ = rank;
  rank_ = rank + node_rank * num_ranks / world_size;
  num_ranks_ = num_ranks;
  world_size_ = world_size;
  node_rank_ = node_rank;
  hidden_size_ = hidden_size;
  num_experts_ = num_experts;
  KLLM_LOG_INFO << fmt::format(
      "DeepEPWrapper::DeepEPWrapper start rank = {}, num_ranks = {}, world_size = {}, node_rank = {}, hidden_size = "
      "{}, num_experts = {}",
      rank_, num_ranks_, world_size_, node_rank_, hidden_size_, num_experts_);
  InitConfig();

  // Each executor holds one DeepEP::Buffer
  bool low_latency_mode = false;
  size_t hidden_bytes = hidden_size_ * 2;
  size_t num_nvl_bytes = std::max(dispatch_config_->get_nvl_buffer_size_hint(hidden_bytes, num_ranks_),
                                  combine_config_->get_nvl_buffer_size_hint(hidden_bytes, num_ranks_));
  size_t num_rdma_bytes = 0;
  if (num_ranks_ > kNumMaxNvlPeers || low_latency_mode) {
    num_rdma_bytes =
        std::max(dispatch_config_->get_rdma_buffer_size_hint(static_cast<int64_t>(hidden_bytes), num_ranks_),
                 combine_config_->get_rdma_buffer_size_hint(static_cast<int64_t>(hidden_bytes), num_ranks_));
  }
  buffer_ = std::make_shared<expert_parallel::deepep::Buffer>(rank_, num_ranks_, static_cast<int64_t>(num_nvl_bytes),
                                                              static_cast<int64_t>(num_rdma_bytes), low_latency_mode,
                                                              c10::cuda::getCurrentCUDAStream(local_rank_));
  if (buffer_->get_num_rdma_ranks() > 1) {
    // Multi-node scenario requires configuring nvshmem environment variables
    std::vector<std::pair<std::string, std::string>> env_vars = {{"NVSHMEM_DISABLE_P2P", "0"},
                                                                 {"NVSHMEM_IB_ENABLE_IBGDA", "1"},
                                                                 {"NVSHMEM_IBGDA_NUM_RC_PER_PE", "24"},
                                                                 {"NVSHMEM_QP_DEPTH", "1024"},
                                                                 {"NVSHMEM_MAX_TEAMS", "7"},
                                                                 {"NVSHMEM_DISABLE_NVLS", "1"},
                                                                 {"NVSHMEM_CUMEM_GRANULARITY", "536870912"},
                                                                 {"NVSHMEM_DISABLE_MNNVL", "1"}};
    for (const auto& env_var : env_vars) {
      if (setenv(env_var.first.c_str(), env_var.second.c_str(), 1) != 0) {
        KLLM_LOG_WARNING << "Failed to set " << env_var.first;
      } else {
        KLLM_LOG_INFO << "Set " << env_var.first << "=" << env_var.second;
      }
    }
  }
  KLLM_LOG_INFO << fmt::format("DeepEPWrapper::DeepEPWrapper end (buffer created, not synced yet)");
#else
  KLLM_THROW("DeepEP is not available. Please install DeepEP library to /usr/local/deepep/ and rebuild.");
#endif
}

EPCommunicationConfig DeepEPWrapper::GetCommunicationConfig() {
#ifdef KSANA_WITH_DEEPEP
  EPCommunicationConfig comm_config;

  // Get IPC Handle address for intra-node card-to-card exchange
  std::string ipc_handle = buffer_->get_local_ipc_handle();
  std::memcpy(comm_config.ipc_handles, ipc_handle.data(), CUDA_IPC_HANDLE_SIZE);

  // Get nvshmem_unique_id for inter-node card-to-card exchange
  auto unique_id = deep_ep::internode::get_unique_id();
  std::memcpy(comm_config.nvshmem_id, unique_id.data(), sizeof(nvshmemx_uniqueid_t));
  return comm_config;
#else
  KLLM_THROW("DeepEP is not available. GetCommunicationConfig() cannot be called.");
#endif
}

DeepEPWrapper::~DeepEPWrapper() {
#ifdef KSANA_WITH_DEEPEP
  if (buffer_->get_num_rdma_ranks() > 1) {
    nvshmem_finalize();
  }
#endif
}

void DeepEPWrapper::Initialize(const std::vector<EPCommunicationConfig>& comm_configs) {
#ifdef KSANA_WITH_DEEPEP
  buffer_->sync(comm_configs);
  // Initialize nvshmem barrier if using RDMA
  if (buffer_->get_num_rdma_ranks() > 1) {
    nvshmem_barrier_all();
  }
#else
  KLLM_THROW("DeepEP is not available. Initialize() cannot be called.");
#endif
}

void DeepEPWrapper::InitConfig() {
#ifdef KSANA_WITH_DEEPEP
  // Adapted from DeepEP [https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py#L218]
  std::unordered_map<int, std::vector<int>> dispatch_config_map = {
      {2, {24, 256, 6, 128}},    {4, {6, 256, 6, 128}},    {8, {6, 256, 6, 128}},    {16, {36, 288, 20, 128}},
      {24, {8, 288, 32, 128}},   {32, {32, 288, 32, 128}}, {64, {20, 288, 28, 128}}, {128, {20, 560, 32, 128}},
      {144, {32, 720, 12, 128}}, {160, {28, 720, 12, 128}}};
  // TODO(zezhao): 后续修改为可动态配置的
  constexpr int kDeepEPDefaultNumSMs = 20;
  num_sms_ = kDeepEPDefaultNumSMs;
  std::vector<int> dispatch_config_param = dispatch_config_map[num_ranks_];
  dispatch_config_ = std::make_shared<deep_ep::Config>(num_sms_, dispatch_config_param[0], dispatch_config_param[1],
                                                       dispatch_config_param[2], dispatch_config_param[3]);

  // Adapted from DeepEP [https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py#L246]
  std::unordered_map<int, std::vector<int>> combine_config_map = {
      {2, {10, 256, 6, 128}},  {4, {9, 256, 6, 128}},  {8, {4, 256, 6, 128}},   {16, {4, 288, 12, 128}},
      {24, {1, 288, 8, 128}},  {32, {1, 288, 8, 128}}, {64, {1, 288, 20, 128}}, {128, {1, 560, 12, 128}},
      {144, {2, 720, 8, 128}}, {160, {2, 720, 8, 128}}};
  std::vector<int> combine_config_param = combine_config_map[num_ranks_];
  combine_config_ = std::make_shared<deep_ep::Config>(num_sms_, combine_config_param[0], combine_config_param[1],
                                                      combine_config_param[2], combine_config_param[3]);
#else
  KLLM_THROW("DeepEP is not available. InitConfig() cannot be called.");
#endif
}

void DeepEPWrapper::SetNumSMs(int new_num_sms) {
#ifdef KSANA_WITH_DEEPEP
  num_sms_ = new_num_sms;
#else
  KLLM_THROW("DeepEP is not available. SetNumSMs() cannot be called.");
#endif
}

#ifdef KSANA_WITH_DEEPEP
void DeepEPWrapper::DispatchLayout(torch::Tensor& topk_ids) {
  auto dispatch_layout_result = buffer_->get_dispatch_layout(topk_ids, num_experts_, empty_event_, false, false);
  num_tokens_per_rank_ = std::get<0>(dispatch_layout_result);
  auto opt_num_tokens_per_rdma_rank = std::get<1>(dispatch_layout_result);
  if (opt_num_tokens_per_rdma_rank.has_value()) {
    num_tokens_per_rdma_rank_ = opt_num_tokens_per_rdma_rank.value();
  }
  num_tokens_per_expert_ = std::get<2>(dispatch_layout_result);
  is_token_in_rank_ = std::get<3>(dispatch_layout_result);
}

void DeepEPWrapper::IntraNodeDispatch(const torch::Tensor& hidden_states,
                                      const std::optional<torch::Tensor>& hidden_states_scales, torch::Tensor& topk_ids,
                                      torch::Tensor& topk_weights) {
  int expert_alignment = 1;
  int num_worst_tokens = 0;
  auto result =
      buffer_->intranode_dispatch(hidden_states,                               // const at::Tensor&
                                  hidden_states_scales,                        // const std::optional<at::Tensor>&
                                  std::make_optional(topk_ids),                // const std::optional<at::Tensor>&
                                  std::make_optional(topk_weights),            // const std::optional<at::Tensor>&
                                  std::make_optional(num_tokens_per_rank_),    // const std::optional<at::Tensor>&
                                  is_token_in_rank_,                           // const at::Tensor&
                                  std::make_optional(num_tokens_per_expert_),  // const std::optional<at::Tensor>&
                                  0,                                           // int
                                  std::nullopt,                                // const std::optional<at::Tensor>&
                                  std::nullopt,                                // const std::optional<at::Tensor>&
                                  expert_alignment,                            // int (不是引用!)
                                  num_worst_tokens,                            // int (不是引用!)
                                  *dispatch_config_,                           // const deep_ep::Config&
                                  empty_event_,                                // std::optional<deep_ep::EventHandle>&
                                  false,                                       // bool
                                  false, output_);

  // Return values that are not used in subsequent computations:
  // 4: num_recv_tokens_per_expert_list
  // 6: channel_prefix_matrix
  // 10: event
  recv_num_tokens_ = std::get<0>(result);
  auto opt_recv_hidden_states_scales = std::get<1>(result);
  if (opt_recv_hidden_states_scales.has_value()) {
    hidden_states_scales_ = opt_recv_hidden_states_scales;
  }
  auto opt_recv_topk_ids = std::get<2>(result);
  if (opt_recv_topk_ids.has_value()) {
    topk_ids_ = opt_recv_topk_ids.value();
  }
  auto opt_recv_topk_weights = std::get<3>(result);
  if (opt_recv_topk_weights.has_value()) {
    topk_weights_ = opt_recv_topk_weights.value();
  }
  rank_prefix_matrix_ = std::get<5>(result);
  channel_prefix_matrix_ = std::get<7>(result);
  src_idx_ = std::get<8>(result);
  send_head_ = std::get<9>(result);
}

void DeepEPWrapper::InterNodeDispatch(const torch::Tensor& hidden_states,
                                      const std::optional<torch::Tensor>& hidden_states_scales, torch::Tensor& topk_ids,
                                      torch::Tensor& topk_weights) {
  int expert_alignment = 1;
  auto result =
      buffer_->internode_dispatch(hidden_states,                                  // const at::Tensor&
                                  hidden_states_scales,                           // const std::optional<at::Tensor>&
                                  std::make_optional(topk_ids),                   // const std::optional<at::Tensor>&
                                  std::make_optional(topk_weights),               // const std::optional<at::Tensor>&
                                  std::make_optional(num_tokens_per_rank_),       // const std::optional<at::Tensor>&
                                  std::make_optional(num_tokens_per_rdma_rank_),  // const std::optional<at::Tensor>&
                                  is_token_in_rank_,                              // const at::Tensor&
                                  std::make_optional(num_tokens_per_expert_),     // const std::optional<at::Tensor>&
                                  0,                                              // int
                                  0,                                              // int
                                  std::nullopt,                                   // const std::optional<at::Tensor>&
                                  std::nullopt,                                   // const std::optional<at::Tensor>&
                                  std::nullopt,                                   // const std::optional<at::Tensor>&
                                  std::nullopt,                                   // const std::optional<at::Tensor>&
                                  expert_alignment,                               // int
                                  *dispatch_config_,                              // const deep_ep::Config&
                                  empty_event_,  // std::optional<deep_ep::EventHandle>&
                                  false,         // bool
                                  false, output_);
  // 0: recv_num_tokens_
  // 1: recv_hidden_states_scales
  // 2: recv_topk_idx
  // 3: recv_topk_weights
  // 4: num_recv_tokens_per_expert_list
  // 5: rdma_channel_prefix_matrix
  // 6: gbl_channel_prefix_matrix
  // 7: recv_rdma_channel_prefix_matrix
  // 8: recv_rdma_rank_prefix_sum
  // 9: recv_gbl_channel_prefix_matrix
  // 10: recv_gbl_rank_prefix_sum
  // 11: recv_src_meta
  // 12: send_rdma_head
  // 13: send_nvl_head
  // 14: event

  // Return values that are not used in subsequent computations:
  // 4: num_recv_tokens_per_expert_list
  // 5: rdma_channel_prefix_matrix
  // 6: gbl_channel_prefix_matrix
  // 14: event
  recv_num_tokens_ = std::get<0>(result);
  auto opt_recv_hidden_states_scales = std::get<1>(result);
  if (opt_recv_hidden_states_scales.has_value()) {
    hidden_states_scales_ = opt_recv_hidden_states_scales;
  }
  auto opt_recv_topk_ids = std::get<2>(result);
  if (opt_recv_topk_ids.has_value()) {
    topk_ids_ = opt_recv_topk_ids.value();
  }
  auto opt_recv_topk_weights = std::get<3>(result);
  if (opt_recv_topk_weights.has_value()) {
    topk_weights_ = opt_recv_topk_weights.value();
  }

  auto opt_recv_rdma_channel_prefix_matrix = std::get<7>(result);
  if (opt_recv_rdma_channel_prefix_matrix.has_value()) {
    recv_rdma_channel_prefix_matrix_ = opt_recv_rdma_channel_prefix_matrix.value();
  }
  recv_rdma_rank_prefix_sum_ = std::get<8>(result);
  auto opt_recv_gbl_channel_prefix_matrix = std::get<9>(result);
  if (opt_recv_gbl_channel_prefix_matrix.has_value()) {
    recv_gbl_channel_prefix_matrix_ = opt_recv_gbl_channel_prefix_matrix.value();
  }

  recv_gbl_rank_prefix_sum_ = std::get<10>(result);

  auto opt_recv_src_meta = std::get<11>(result);
  if (opt_recv_src_meta.has_value()) {
    recv_src_meta_ = opt_recv_src_meta.value();
  }
  auto opt_send_rdma_head = std::get<12>(result);
  if (opt_send_rdma_head.has_value()) {
    send_rdma_head_ = opt_send_rdma_head.value();
  }
  auto opt_send_nvl_head = std::get<13>(result);
  if (opt_send_nvl_head.has_value()) {
    send_nvl_head_ = opt_send_nvl_head.value();
  }
}
#endif  // KSANA_WITH_DEEPEP

void DeepEPWrapper::Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                             Stream stream) {
#ifdef KSANA_WITH_DEEPEP
  // input_tensors:
  //   0: hidden_states (num_tokens, hidden_size)
  //   1: topk_ids (num_tokens, topk)
  //   2: topk_weights (num_tokens, topk)
  //   [optional] 3: a1_q (num_tokens, hidden_size) - FP8量化后的hidden_states
  //   [optional] 4: a1_scale (num_tokens, hidden_size/128) - FP8量化的scale
  //   [optional] 5: workspace - 工作空间
  KLLM_LOG_DEBUG << fmt::format("DeepEPWrapper::Dispatch start, input_tensors.size()={}", input_tensors.size());

  // Get cudaStream_t for DeepEP call
  cudaStream_t cuda_stream = stream.Get();

  // Set input buffer
  bool use_fp8_quant = input_tensors.size() > 3;
  if (use_fp8_quant) {
    hidden_states_ = TransToTorchTensor(input_tensors[3]);         // a1_q
    hidden_states_scales_ = TransToTorchTensor(input_tensors[4]);  // a1_scale
  } else {
    hidden_states_ = TransToTorchTensor(input_tensors[0]);
    hidden_states_scales_ = std::nullopt;
  }

  // topk_ids need to be int64
  auto topk_ids_int32 = TransToTorchTensor(input_tensors[1]);
  topk_ids_ = topk_ids_int32.to(torch::kInt64);

  topk_weights_ = TransToTorchTensor(input_tensors[2]);

  // Set output buffer
  if (use_fp8_quant && input_tensors.size() > 5) {
    output_ = input_tensors[5].GetPtr<void>();
  } else {
    output_ = output_tensors[1].GetPtr<void>();
  }

  // Step 1: Execute dispatch_layout
  DispatchLayout(topk_ids_);

  // Step 2: Execute Dispatch
  if (buffer_->get_num_rdma_ranks() > 1) {
    InterNodeDispatch(hidden_states_, hidden_states_scales_, topk_ids_, topk_weights_);
  } else {
    IntraNodeDispatch(hidden_states_, hidden_states_scales_, topk_ids_, topk_weights_);
  }

  // Step 3: Copy output to output_tensors
  size_t hidden_size = input_tensors[0].shape[1];
  if (recv_num_tokens_ > 0) {
    if (use_fp8_quant) {
      cudaMemcpyAsync(output_tensors[2].GetPtr<int>(), output_, recv_num_tokens_ * hidden_size,
                      cudaMemcpyDeviceToDevice, cuda_stream);
      cudaMemcpyAsync(output_tensors[3].GetPtr<int>(), hidden_states_scales_.value().data_ptr(),
                      hidden_states_scales_.value().numel() * hidden_states_scales_.value().element_size(),
                      cudaMemcpyDeviceToDevice, cuda_stream);
    }
    auto topk_ids_int32 = topk_ids_.to(torch::kInt32);
    cudaMemcpyAsync(input_tensors[1].GetPtr<int>(), topk_ids_int32.data_ptr(),
                    topk_ids_int32.numel() * topk_ids_int32.element_size(), cudaMemcpyDeviceToDevice, cuda_stream);
    cudaMemcpyAsync(input_tensors[2].GetPtr<float>(), topk_weights_.data_ptr(),
                    topk_weights_.numel() * topk_weights_.element_size(), cudaMemcpyDeviceToDevice, cuda_stream);
  }

  // Step 4: change output_tensors shape
  // output_tensors[1]: dispatched_hidden_states (recv_num_tokens_, hidden_size)
  output_tensors[1].shape[0] = recv_num_tokens_;
  output_tensors[1].shape[1] = hidden_size;
  KLLM_LOG_DEBUG << fmt::format("DeepEPWrapper::Dispatch end, recv_num_tokens_={}", recv_num_tokens_);
#else
  KLLM_THROW("DeepEP is not available. Dispatch() cannot be called.");
#endif
}

#ifdef KSANA_WITH_DEEPEP
void DeepEPWrapper::IntraNodeCombine() {
  auto result = buffer_->intranode_combine(hidden_states_, std::nullopt, std::nullopt, std::nullopt, src_idx_,
                                           rank_prefix_matrix_, channel_prefix_matrix_, send_head_, *combine_config_,
                                           empty_event_, false, false, output_);
  recv_num_tokens_ = std::get<0>(result);
}

void DeepEPWrapper::InterNodeCombine() {
  auto result = buffer_->internode_combine(hidden_states_, std::nullopt, std::nullopt, std::nullopt, recv_src_meta_,
                                           is_token_in_rank_, recv_rdma_channel_prefix_matrix_,
                                           recv_rdma_rank_prefix_sum_, recv_gbl_channel_prefix_matrix_, send_rdma_head_,
                                           send_nvl_head_, *combine_config_, empty_event_, false, false, output_);
  recv_num_tokens_ = std::get<0>(result);
}
#endif  // KSANA_WITH_DEEPEP

void DeepEPWrapper::Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                            Stream stream) {
#ifdef KSANA_WITH_DEEPEP
  // input_tensors:
  //   0: expert_output (recv_num_tokens_, hidden_size) - 专家计算后的输出
  // output_tensors:
  //   0: combined_output (original_num_tokens, hidden_size) - 合并后的输出
  KLLM_LOG_DEBUG << fmt::format("DeepEPWrapper::Combine start, input_tensors.size()={}", input_tensors.size());

  // Get cudaStream_t for DeepEP call
  cudaStream_t cuda_stream = stream.Get();

  // Prepare input data
  void* data_ptr = input_tensors[0].GetPtr<void>();
  std::vector<int64_t> shape_vec(input_tensors[0].shape.begin(), input_tensors[0].shape.end());
  // DeepEP only support torch::kBFloat16
  auto bfp16_options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
  hidden_states_ = torch::from_blob(data_ptr, shape_vec, bfp16_options);

  // Set output buffer
  output_ = output_tensors[0].GetPtr<void>();

  // Execute Combine
  if (buffer_->get_num_rdma_ranks() > 1) {
    InterNodeCombine();
  } else {
    IntraNodeCombine();
  }
  KLLM_LOG_DEBUG << fmt::format("DeepEPWrapper::Combine end");
#else
  KLLM_THROW("DeepEP is not available. Combine() cannot be called.");
#endif
}

}  // namespace ksana_llm
