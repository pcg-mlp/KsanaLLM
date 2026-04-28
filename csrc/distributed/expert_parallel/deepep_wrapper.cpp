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

DeepEPWrapper::DeepEPWrapper(int rank, int num_ranks, int world_size, int node_rank, int hidden_size, int num_experts,
                             bool low_latency_mode, int max_token_num) {
#ifdef KSANA_WITH_DEEPEP
  local_rank_ = rank;
  rank_ = rank + node_rank * num_ranks / world_size;
  num_ranks_ = num_ranks;
  world_size_ = world_size;
  node_rank_ = node_rank;
  hidden_size_ = hidden_size;
  num_experts_ = num_experts;
  low_latency_mode_ = low_latency_mode;

  // Calculate num_max_dispatch_tokens_ for low latency mode
  // This represents the total maximum tokens across all ranks
  if (low_latency_mode_) {
    num_max_dispatch_tokens_ = max_token_num;
    KLLM_LOG_INFO << fmt::format("Low latency mode enabled, num_max_dispatch_tokens_: {}", num_max_dispatch_tokens_);
  }
  InitConfig();

  // Each executor holds one DeepEP::Buffer
  constexpr size_t kBF16Bytes = 2;
  size_t hidden_bytes = hidden_size_ * kBF16Bytes;
  size_t num_nvl_bytes = 0;
  size_t num_rdma_bytes = 0;

  if (low_latency_mode_) {
    // Low latency mode requires only RDMA buffers
    // RDMA buffer: use get_low_latency_rdma_size_hint for low latency mode
    num_rdma_bytes =
        deep_ep::get_low_latency_rdma_size_hint(num_max_dispatch_tokens_, hidden_size_, num_ranks_, num_experts_);
  } else {
    // Normal mode uses NVL and optionally RDMA for multi-node
    num_nvl_bytes = dispatch_config_->get_nvl_buffer_size_hint(static_cast<int64_t>(hidden_bytes), num_ranks_) +
                    combine_config_->get_nvl_buffer_size_hint(static_cast<int64_t>(hidden_bytes), num_ranks_);
    if (num_ranks_ > kNumMaxNvlPeers) {
      num_rdma_bytes =
          std::max(dispatch_config_->get_rdma_buffer_size_hint(static_cast<int64_t>(hidden_bytes), num_ranks_),
                   combine_config_->get_rdma_buffer_size_hint(static_cast<int64_t>(hidden_bytes), num_ranks_));
    }
  }

  KLLM_LOG_INFO << fmt::format("DeepEP allocating NVL buffer = {} MB, RDMA buffer = {} MB", num_nvl_bytes / 1e6,
                               num_rdma_bytes / 1e6);

  buffer_ = std::make_shared<expert_parallel::deepep::Buffer>(rank_, num_ranks_, static_cast<int64_t>(num_nvl_bytes),
                                                              static_cast<int64_t>(num_rdma_bytes), low_latency_mode_);
  if (buffer_->get_num_rdma_ranks() > 1 || low_latency_mode_) {
    int num_qps_per_rank = low_latency_mode_ ? (num_experts_ / num_ranks_) : 24;
    // Multi-node scenario requires configuring nvshmem environment variables
    std::vector<std::pair<std::string, std::string>> env_vars = {
        {"NVSHMEM_DISABLE_P2P", "0"},
        {"NVSHMEM_IB_ENABLE_IBGDA", "1"},
        {"NVSHMEM_IBGDA_NUM_RC_PER_PE", std::to_string(num_qps_per_rank)},
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
  KLLM_CHECK_WITH_INFO(buffer_->is_available(), "DeepEP buffer is not available.");
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

void DeepEPWrapper::LowLatencyDispatch(const torch::Tensor& hidden_states, torch::Tensor& topk_ids) {
  int num_tokens = hidden_states.size(0);

  bool use_fp8 = (hidden_states.scalar_type() == torch::kFloat8_e4m3fn);
  std::optional<torch::Tensor> fp8_quant_scale = std::nullopt;
  bool use_ue8m0 = false;
  bool async = false;
  bool return_recv_hook = false;

  auto result = buffer_->low_latency_dispatch(hidden_states, topk_ids, num_max_dispatch_tokens_, num_experts_, use_fp8,
                                              fp8_quant_scale, async, return_recv_hook, output_);
  // Result tuple: (recv_hidden_states, opt_scales, packed_recv_count, src_info, layout_range, event, hook)
  recv_num_tokens_ = std::get<0>(result);
  auto opt_scales = std::get<1>(result);
  low_latency_packed_recv_count_ = std::get<2>(result);
  low_latency_src_info_ = std::get<3>(result);
  low_latency_layout_range_ = std::get<4>(result);
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
  // Get cudaStream_t for DeepEP call
  cudaStream_t cuda_stream = stream.Get();

  // CRITICAL FIX: Set PyTorch's current CUDA stream to KsanaLLM's compute stream
  // DeepEP internally calls at::cuda::getCurrentCUDAStream() to get compute_stream,
  // which defaults to PyTorch's default stream (nullptr/0x0) if not explicitly set.
  // This causes cudaEventRecord on a nullptr stream in stream_wait(), leading to crash.
  c10::cuda::CUDAStream torch_compute_stream = c10::cuda::getStreamFromExternal(cuda_stream, local_rank_);
  c10::cuda::setCurrentCUDAStream(torch_compute_stream);

  // Directly create torch tensor from input without intermediate buffer copy
  bool use_fp8_quant = input_tensors.size() > 3;

  if (use_fp8_quant) {
    hidden_states_ = TransToTorchTensor(input_tensors[3]);         // a1_q (FP8 quantized)
    hidden_states_scales_ = TransToTorchTensor(input_tensors[4]);  // a1_scale
  } else {
    hidden_states_ = TransToTorchTensor(input_tensors[0]);  // hidden_states
    // DeepEP only supports BFloat16, convert FP16 to BFloat16 if necessary
    if (hidden_states_.scalar_type() == torch::kFloat16) {
      KLLM_LOG_DEBUG << "Converting hidden_states from FP16 to BFloat16 for DeepEP";
      hidden_states_ = hidden_states_.to(torch::kBFloat16);
    }
    hidden_states_scales_ = std::nullopt;
  }

  // topk_ids need to be int64
  auto topk_ids_int32 = TransToTorchTensor(input_tensors[1]);
  topk_ids_ = topk_ids_int32.to(torch::kInt64);

  topk_weights_ = TransToTorchTensor(input_tensors[2]);

  // Execute Dispatch based on mode
  if (low_latency_mode_) {
    // Low latency mode: dispatch directly returns the dispatched tokens
    output_ = output_tensors[1].GetPtr<void>();
    LowLatencyDispatch(hidden_states_, topk_ids_);

    // Update output tensor shape
    size_t hidden_size = input_tensors[0].shape[1];

    // Low latency mode uses fixed 3D shape: [num_experts_per_node, max_tokens_per_expert, hidden_size]
    // This is different from Normal mode which uses dynamic 2D shape: [recv_num_tokens_, hidden_size]
    output_tensors[1].shape.resize(3);
    output_tensors[1].shape[0] = num_experts_ / num_ranks_;  // num_experts_per_node
    output_tensors[1].shape[1] = num_max_dispatch_tokens_ * num_ranks_;
    output_tensors[1].shape[2] = hidden_size;

    // Copy packed_recv_count to output_tensors (if provided by moe_layer)
    // output_tensors[2] is the ll_recv_count_ buffer allocated in moe_layer
    if (output_tensors.size() > 2) {
      // Copy packed_recv_count from low_latency_packed_recv_count_ to the target address
      cudaMemcpyAsync(output_tensors[2].GetPtr<int>(), low_latency_packed_recv_count_.data_ptr(),
                      low_latency_packed_recv_count_.numel() * low_latency_packed_recv_count_.element_size(),
                      cudaMemcpyDeviceToDevice, cuda_stream);
      output_tensors[2].shape = {num_experts_ / num_ranks_};
    }

    // Copy scales if using FP8 quantization (different buffer)
    if (use_fp8_quant && hidden_states_scales_.has_value()) {
      cudaMemcpyAsync(output_tensors[3].GetPtr<int>(), hidden_states_scales_.value().data_ptr(),
                      hidden_states_scales_.value().numel() * hidden_states_scales_.value().element_size(),
                      cudaMemcpyDeviceToDevice, cuda_stream);
    }
  } else {
    // Standard mode dispatch
    // Step 1: Execute dispatch_layout
    DispatchLayout(topk_ids_);

    // Step 2: Execute Dispatch - output directly to output_tensors[1]
    output_ = output_tensors[1].GetPtr<void>();
    if (buffer_->get_num_rdma_ranks() > 1) {
      InterNodeDispatch(hidden_states_, hidden_states_scales_, topk_ids_, topk_weights_);
    } else {
      IntraNodeDispatch(hidden_states_, hidden_states_scales_, topk_ids_, topk_weights_);
    }

    // Step 3: Copy metadata to output_tensors (no need to copy hidden_states as it's already in output_tensors[1])
    size_t hidden_size = input_tensors[0].shape[1];
    if (recv_num_tokens_ > 0) {
      if (use_fp8_quant && hidden_states_scales_.has_value()) {
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
  }

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

void DeepEPWrapper::LowLatencyCombine(const torch::Tensor& expert_output, torch::Tensor& topk_ids,
                                      torch::Tensor& topk_weights) {
  bool zero_copy = false;
  bool async = false;
  bool return_recv_hook = false;
  bool use_logfmt = false;

  auto result = buffer_->low_latency_combine(expert_output, topk_ids, topk_weights, low_latency_src_info_,
                                             low_latency_layout_range_, num_max_dispatch_tokens_, num_experts_,
                                             use_logfmt, zero_copy, async, return_recv_hook, output_);
  recv_num_tokens_ = std::get<0>(result);
}
#endif  // KSANA_WITH_DEEPEP

void DeepEPWrapper::Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                            Stream stream) {
#ifdef KSANA_WITH_DEEPEP
  // input_tensors:
  //   0: expert_output (recv_num_tokens_, hidden_size) - 专家计算后的输出
  //   0(low latency mode): expert_output (num_experts_per_node, max_tokens_per_expert, hidden_size) - 专家计算后的输出
  // output_tensors:
  //   0: combined_output (original_num_tokens, hidden_size) - 合并后的输出
  // Get cudaStream_t for DeepEP call
  cudaStream_t cuda_stream = stream.Get();
  c10::cuda::CUDAStream torch_compute_stream = c10::cuda::getStreamFromExternal(cuda_stream, local_rank_);
  c10::cuda::setCurrentCUDAStream(torch_compute_stream);

  std::vector<int64_t> shape_vec(input_tensors[0].shape.begin(), input_tensors[0].shape.end());

  // Track original dtype to determine if we need to convert back after combine
  bool original_is_fp16 = (input_tensors[0].dtype == TYPE_FP16);

  // Create tensor directly from input data
  hidden_states_ = TransToTorchTensor(input_tensors[0]);

  // DeepEP only supports BFloat16, convert if necessary
  if (hidden_states_.scalar_type() == torch::kFloat16) {
    hidden_states_ = hidden_states_.to(torch::kBFloat16);
  }

  output_ = output_tensors[0].GetPtr<void>();

  // Execute Combine based on mode
  if (low_latency_mode_) {
    // Low latency mode: combine uses the stored src_info and layout_range from dispatch
    LowLatencyCombine(hidden_states_, topk_ids_, topk_weights_);

    // Convert back to original dtype if it was FP16
    // If original was BFP16, no conversion needed (hidden_states_ is already BFP16 from LowLatencyCombine)
    if (original_is_fp16) {
      hidden_states_ = hidden_states_.to(torch::kFloat16);
    }

    KLLM_LOG_DEBUG << fmt::format("Combine: hidden_states_ shape=[{}, {}], numel={}, element_size={}, data_ptr={:p}",
                                  hidden_states_.size(0), hidden_states_.size(1), hidden_states_.numel(),
                                  hidden_states_.element_size(), hidden_states_.data_ptr());
  } else {
    // Standard mode combine - output directly to final buffer
    output_ = output_tensors[0].GetPtr<void>();

    if (buffer_->get_num_rdma_ranks() > 1) {
      InterNodeCombine();
    } else {
      IntraNodeCombine();
    }
  }

  KLLM_LOG_DEBUG << fmt::format("DeepEPWrapper::Combine end");
#else
  KLLM_THROW("DeepEP is not available. Combine() cannot be called.");
#endif
}

}  // namespace ksana_llm
