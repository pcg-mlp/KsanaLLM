/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/cutlass_moe_layer.h"

#include "configure/environment.h"
#include "csrc/kernels/nvidia/moe/cutlass_moe/low_latency_adapter.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/utils.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "layers/grouped_topk_layer.h"
#include "profiler/timer.h"
#include "utils/search_status.h"
#include "utils/singleton.h"
#include "utils/utils.h"

namespace ksana_llm {

#define TRTLLM_CUTLASS_MOE_CHUNK_SIZE ((size_t)(8 * 1024))

// 计算n的位宽（最高有效位的位置+1）
inline size_t bit_width(size_t n) {
  size_t width = 0;
  while (n > 0) {
    n >>= 1;
    ++width;
  }
  return width;
}

// 计算bit_floor(n)的位宽，即floor(log2(n)) + 1
inline size_t bit_width_of_bit_floor(size_t n) { return bit_width(n); }

// 计算bit_ceil(n)的位宽，即ceil(log2(n)) + 1
inline size_t bit_width_of_bit_ceil(size_t n) {
  if (n <= 1) return 1;
  return bit_width(n - 1) + 1;
}

// dispatch函数：默认使用floor，在特定区间使用ceil
inline size_t bit_width_dispatch(size_t n) {
  // 基于deepseek-r1在当前搜索策略下profile出来的区间
  if (384 < n && n <= 512) {
    return bit_width_of_bit_ceil(n);
  } else if (640 < n && n <= 768) {
    return bit_width_of_bit_ceil(n);
  } else if (896 < n && n <= 1024) {
    return bit_width_of_bit_ceil(n);
  } else {
    return bit_width_of_bit_floor(n);
  }
}

Status CutlassMoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  inter_data_type_ = runtime_config.inter_data_type;
  DISPATCH_BY_3_DTYPE(inter_data_type_, InitT, parameters, runtime_config, context, rank);
}

template <typename T>
Status CutlassMoeLayer::InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                              std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;

  int parameter_index = 0;
  moe_scale_norm_mode_ = std::any_cast<const MoeScaleNormMode>(parameters[parameter_index++]);
  max_token_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  layer_idx_ = std::any_cast<int>(parameters[parameter_index++]);
  expert_num_per_node_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_hidden_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_inter_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_topk_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  tp_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  use_vllm_moe_ = std::any_cast<bool>(parameters[parameter_index++]);
  num_expert_group_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  expert_groups_topk_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  scoring_func_ = std::any_cast<std::string>(parameters[parameter_index++]);
  topk_method_ = std::any_cast<std::string>(parameters[parameter_index++]);
  norm_topk_prob_ = std::any_cast<bool>(parameters[parameter_index++]);
  routed_scaling_factor_ = std::any_cast<float>(parameters[parameter_index++]);
  use_e_score_correction_bias_ = std::any_cast<bool>(parameters[parameter_index++]);
  enable_full_shared_expert_ = std::any_cast<bool>(parameters[parameter_index++]);
  DataType fp8_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  DataType int_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  group_size_ = std::any_cast<int>(parameters[parameter_index++]);
  apply_weight_ = std::any_cast<bool>(parameters[parameter_index++]);
  swiglu_limit_ = std::any_cast<float>(parameters[parameter_index++]);

  if (swiglu_limit_ > 0.0f) {
    swiglu_limit_host_.assign(expert_num_per_node_, swiglu_limit_);
  }

  // 清空tactic
  config_map_.clear();

  // 强制要求TP=EP
  if (runtime_config.parallel_basic_config.tensor_parallel_size !=
      runtime_config.parallel_basic_config.expert_parallel_size) {
    KLLM_THROW("Currently, CutlassMoeLayer strictly enforces TP=EP.");
  }

  // 计算全局ep_size和ep_rank
  global_expert_para_size_ = runtime_config.parallel_basic_config.expert_parallel_size *
                             runtime_config.parallel_basic_config.expert_world_size;
  global_expert_para_rank_ =
      context_->global->GetExpertParallelExpertNodeRank() * runtime_config.parallel_basic_config.expert_parallel_size +
      rank_;

  // 从 runtime_config 获取 MoE all-to-all 通信模式
  using_deepep_ = (runtime_config.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_NORMAL ||
                   runtime_config.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_LOW_LATENCY);
  use_low_latency_mode_ = (runtime_config.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_LOW_LATENCY);
  if (use_low_latency_mode_) {
    low_latency_max_tokens_ = runtime_config.deepep_low_latency_max_tokens;
  }

  // 初始化 GroupedTopkLayer
  grouped_topk_layer_ = std::make_shared<GroupedTopkLayer>();
  // 根据配置文件 moe_topk_score_threshold 设置 score 阈值
  float score_threshold = runtime_config.moe_topk_score_threshold;
  GroupedTopkLayerParameters grouped_topk_params{.topk = expert_topk_,
                                                 .renormalize = norm_topk_prob_,
                                                 .num_expert_group = num_expert_group_,
                                                 .topk_group = expert_groups_topk_,
                                                 .scoring_func = scoring_func_,
                                                 .routed_scaling_factor = routed_scaling_factor_,
                                                 .use_e_score_correction_bias = use_e_score_correction_bias_,
                                                 .score_threshold = score_threshold};
  grouped_topk_layer_->Init({grouped_topk_params}, runtime_config, context, rank);

  // 创建cutlass moe算子
  // DeepEP LowLatency模式下，启用alltoall：finalizeMoeRouting不需要将无效token填零
  cutlass_moe_wrapper_ = std::make_shared<llm_kernels::nvidia::CutlassMoeWrapper>(
      1, 0, global_expert_para_size_, global_expert_para_rank_, 1, 0, expert_topk_,
      /*enable_alltoall=*/use_low_latency_mode_);
  cutlass_moe_wrapper_->Init<T>();

  // EPLB data dump is off by default; when enabled, use the path from runtime_config
  enable_dump_eplb_data_ = runtime_config.enable_dump_eplb_data;
  if (enable_dump_eplb_data_) {
    eplb_dump_path_ = runtime_config.dump_eplb_path;
  }

  // When EPLB acceleration is enabled or using deepep, initialize the expert_map_.
  enable_load_eplb_config_ = runtime_config.enable_load_eplb_config;
  if (enable_load_eplb_config_ || using_deepep_) {
    size_t total_expert_num = expert_num_per_node_ * global_expert_para_size_;
    expert_map_ = std::make_shared<llm_kernels::nvidia::moe::ExpertMap>(global_expert_para_size_,
                                                                        global_expert_para_rank_, total_expert_num);
  }

  KLLM_LOG_INFO << fmt::format("Rank[{}] CutlassMoeLayer Init", rank_);
  return Status();
}

void CutlassMoeLayer::SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager) {
  expert_parallel_manager_ = expert_parallel_manager;
}

size_t CutlassMoeLayer::GetWorkspaceSize() {
  topk_weights_ptr_size = RoundUp(max_token_num_ * expert_topk_ * sizeof(float), 256);
  topk_ids_ptr_size = RoundUp(max_token_num_ * expert_topk_ * sizeof(int32_t), 256);
  // dummy KTensor (null data) — wrapper only reads has_value() to pick SwigluBias sizing
  std::optional<KTensor> swiglu_limit_for_ws;
  if (swiglu_limit_ > 0.0f) {
    swiglu_limit_for_ws = KTensor(nullptr, {expert_num_per_node_}, KScalarType::Float);
  }
  swiglu_limit_ptr_size_ = (swiglu_limit_ > 0.0f) ? RoundUp(expert_num_per_node_ * sizeof(float), 256) : 0;

  // LowLatency模式需要额外的adapted空间和recv_expert_count空间
  if (use_low_latency_mode_) {
    // LowLatency模式下，Dispatch后的shape为 [local_num_experts, num_ranks * max_token_num, hidden_size]
    size_t max_dispatch_token_num = expert_num_per_node_ * global_expert_para_size_ * low_latency_max_tokens_;

    // kernel workspace 需要按照 LL 模式的 max_dispatch_token_num 和 topk=1 来计算
    // chunk 大小取 min(CHUNK_SIZE, max_dispatch_token_num)
    kernel_workspace_size =
        RoundUp(cutlass_moe_wrapper_->GetWorkspaceSize(std::min(TRTLLM_CUTLASS_MOE_CHUNK_SIZE, max_dispatch_token_num),
                                                       1,  // LL模式下 effective_topk = 1
                                                       expert_num_per_node_, expert_hidden_size_, expert_inter_size_,
                                                       std::nullopt, std::nullopt, swiglu_limit_for_ws),
                256);

    // adapted_topk_weights/ids: [max_dispatch_token_num, topk=1]
    adapted_topk_weights_size_ = RoundUp(max_dispatch_token_num * sizeof(float), 256);
    adapted_topk_ids_size_ = RoundUp(max_dispatch_token_num * sizeof(int32_t), 256);
    recv_expert_count_size_ = RoundUp(expert_num_per_node_ * sizeof(int32_t), 256);

    max_ws_bytes_ = topk_weights_ptr_size + topk_ids_ptr_size + adapted_topk_weights_size_ + adapted_topk_ids_size_ +
                    recv_expert_count_size_ + kernel_workspace_size + swiglu_limit_ptr_size_;

    KLLM_LOG_DEBUG << fmt::format(
        "Rank[{}] LowLatency workspace: max_dispatch_token_num={} "
        "(local_experts={} * num_ranks={} * max_tokens={}), "
        "topk_weights={}, topk_ids={}, adapted_weights={}, adapted_ids={}, recv_count={}, kernel={}, swiglu={}, "
        "total={}",
        rank_, max_dispatch_token_num, expert_num_per_node_, global_expert_para_size_, low_latency_max_tokens_,
        topk_weights_ptr_size, topk_ids_ptr_size, adapted_topk_weights_size_, adapted_topk_ids_size_,
        recv_expert_count_size_, kernel_workspace_size, swiglu_limit_ptr_size_, max_ws_bytes_);
  } else {
    kernel_workspace_size =
        RoundUp(cutlass_moe_wrapper_->GetWorkspaceSize(
                    std::min(TRTLLM_CUTLASS_MOE_CHUNK_SIZE, max_token_num_), expert_topk_, expert_num_per_node_,
                    expert_hidden_size_, expert_inter_size_, std::nullopt, std::nullopt, swiglu_limit_for_ws),
                256);
    max_ws_bytes_ = topk_weights_ptr_size + topk_ids_ptr_size + kernel_workspace_size + swiglu_limit_ptr_size_;
    KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for CutlassMoeLayer", rank_, max_ws_bytes_);
  }

  return max_ws_bytes_;
}

Status CutlassMoeLayer::Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
  const size_t record_iters = GetEnvAsPositiveInt("QUANT_PROFILE", 15);
  if (record_iters == 0) {
    KLLM_LOG_DEBUG << "$QUANT_PROFILE==0, Skipping CutlassMoeLayer Preprocess";
    return Status();
  }
  const size_t warmup_iters = std::max(1UL, record_iters / 3);  // warmup不能为0

  static std::mutex g_mtx;
  std::lock_guard<std::mutex> guard(g_mtx);

  const auto start_time = ProfileTimer::GetCurrentTime();

  // 检查是否可以跳过
  if (Singleton<CutlassMoeSearchStatus>::GetInstance()->IsCutlassMoeScheduleContain(
          expert_topk_, expert_num_per_node_, expert_hidden_size_, expert_inter_size_)) {
    config_map_ = Singleton<CutlassMoeSearchStatus>::GetInstance()->GetCutlassMoeSchedule(
        expert_topk_, expert_num_per_node_, expert_hidden_size_, expert_inter_size_);
    KLLM_LOG_INFO << fmt::format("Reusing Profile CutlassMoeLayer Layer in rank:{}, token=({}~{}),({},{},{},{})", rank_,
                                 1, TRTLLM_CUTLASS_MOE_CHUNK_SIZE, expert_topk_, expert_num_per_node_,
                                 expert_hidden_size_, expert_inter_size_);
    return Status();
  }

  // 创建一份假权重
  Tensor w1_w3_weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                               {expert_num_per_node_, expert_inter_size_ * 2, expert_hidden_size_ / 2});
  Tensor w2_weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                            {expert_num_per_node_, expert_hidden_size_, expert_inter_size_ / 2});
  KTensor fc1_expert_weights = TensorToKTensor(w1_w3_weight);
  KTensor fc2_expert_weights = TensorToKTensor(w2_weight);

  auto get_gemm_best_tactic = [&](size_t num_rows, int64_t gemm_idx, size_t warmup_iters,
                                  size_t profile_iters) -> int64_t {
    // 获取workspace
    size_t profile_workspace_size = cutlass_moe_wrapper_->GetProfileWorkspace(
        fc1_expert_weights, std::nullopt, fc2_expert_weights, std::nullopt, num_rows, gemm_idx, -1, true,
        context_->device->GetPrimaryComputeStream().Get());
    // 开辟workspace
    Tensor profile_workspace = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {profile_workspace_size});
    // 设置workspace
    cutlass_moe_wrapper_->SetProfileWorkspace(profile_workspace.GetPtr<void>(), fc1_expert_weights, std::nullopt,
                                              fc2_expert_weights, std::nullopt, num_rows, gemm_idx, -1, true,
                                              context_->device->GetPrimaryComputeStream().Get());
    // 获取最优tactic
    return cutlass_moe_wrapper_->Profile(fc1_expert_weights, std::nullopt, fc2_expert_weights, std::nullopt, num_rows,
                                         gemm_idx, warmup_iters, profile_iters,
                                         context_->device->GetPrimaryComputeStream().Get());
  };

  config_map_.resize(1);  // NOTE(jinxcwu) 填充0位置占位，有效数据从1位置开始
  for (size_t profile_token = 1; profile_token <= TRTLLM_CUTLASS_MOE_CHUNK_SIZE; profile_token *= 2) {
    int64_t gemm1_tactic = get_gemm_best_tactic(profile_token, 1, warmup_iters, record_iters);
    int64_t gemm2_tactic = get_gemm_best_tactic(profile_token, 2, warmup_iters, record_iters);
    config_map_.push_back({gemm1_tactic, gemm2_tactic});
  }

  Singleton<CutlassMoeSearchStatus>::GetInstance()->AddCutlassMoeSchedule(
      expert_topk_, expert_num_per_node_, expert_hidden_size_, expert_inter_size_, config_map_);

  KLLM_LOG_INFO << fmt::format("Rank[{}] CutlassMoeLayer Preprocess cost time: {} s", rank_,
                               ProfileTimer::GetCurrentTime() - start_time);
  return Status();
}

inline std::vector<int64_t> CutlassMoeLayer::GetBestTactic(const size_t& num_rows) {
  if (config_map_.empty()) {
    return {};
  } else {
    if (num_rows <= TRTLLM_CUTLASS_MOE_CHUNK_SIZE) {
      return config_map_[bit_width_dispatch(num_rows)];
    } else {
      return config_map_.back();  // 超出profile范围，取最后一个tactic
    }
  }
}

inline Status CutlassMoeLayer::ProcessChunks(const Tensor& input_tensor, Tensor& output_tensor,
                                             const size_t total_tokens, const KTensor& fc1_expert_weights_ktensor,
                                             const KTensor& fc2_expert_weights_ktensor,
                                             const std::vector<KTensor>& quant_scales_ktensors) {
  // 在 LowLatency 模式下，使用适配后的 topk_ids 和 topk_weights
  void* topk_weights_to_use = use_low_latency_mode_ ? adapted_topk_weights_ptr_ : topk_weights_ptr_;
  void* topk_ids_to_use = use_low_latency_mode_ ? adapted_topk_ids_ptr_ : topk_ids_ptr_;

  // LowLatency 模式下 topk=1，Normal 模式下 topk=expert_topk_
  const size_t effective_topk = use_low_latency_mode_ ? 1 : expert_topk_;
  for (size_t chunk_token = 0; chunk_token < total_tokens; chunk_token += TRTLLM_CUTLASS_MOE_CHUNK_SIZE) {
    size_t start_token = chunk_token;
    size_t end_token = std::min(chunk_token + TRTLLM_CUTLASS_MOE_CHUNK_SIZE, total_tokens);
    size_t solve_token = end_token - start_token;

    Tensor chunk_input(input_tensor.location, input_tensor.dtype, {solve_token, input_tensor.shape[1]},
                       input_tensor.GetPtr<void>() + start_token * input_tensor.shape[1] * input_tensor.GetDTypeSize());
    Tensor chunk_output(
        output_tensor.location, output_tensor.dtype, {solve_token, output_tensor.shape[1]},
        output_tensor.GetPtr<void>() + start_token * output_tensor.shape[1] * output_tensor.GetDTypeSize());
    KTensor input_ktensor = TensorToKTensor(chunk_input);
    KTensor output_ktensor = TensorToKTensor(chunk_output);

    Tensor chunk_topk_weights(input_tensor.location, TYPE_FP32, {solve_token, effective_topk},
                              topk_weights_to_use + start_token * effective_topk * sizeof(float));
    Tensor chunk_topk_ids(input_tensor.location, TYPE_INT32, {solve_token, effective_topk},
                          topk_ids_to_use + start_token * effective_topk * sizeof(int32_t));
    KTensor token_selected_experts_ktensor = TensorToKTensor(chunk_topk_ids);
    KTensor token_final_scales_ktensor = TensorToKTensor(chunk_topk_weights);

    std::optional<KTensor> swiglu_limit_ktensor;
    if (swiglu_limit_ > 0.0f) {
      swiglu_limit_ktensor = KTensor(swiglu_limit_ptr_, {expert_num_per_node_}, KScalarType::Float);
    }

    // TODO(zezhao): GetBestTactic(solve_token) 用于获取最优tactic，LowLatency模式下由于空桶存在，因此传入的该值过大，
    // 在较低有效 token 数时性能会下降
    size_t valid_tokens = use_low_latency_mode_ ? low_latency_max_tokens_ : solve_token;
    cutlass_moe_wrapper_->Forward(output_ktensor, input_ktensor, token_selected_experts_ktensor,
                                  token_final_scales_ktensor, fc1_expert_weights_ktensor, fc2_expert_weights_ktensor,
                                  quant_scales_ktensors, std::nullopt, std::nullopt, swiglu_limit_ktensor,
                                  GetBestTactic(valid_tokens), context_->device->GetPrimaryComputeStream().Get());
  }
  return Status();
}

Status CutlassMoeLayer::ExecuteGroupedTopk(const Tensor* hidden_states, const Tensor* gating_output,
                                           const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                                           const Tensor* e_score_bias) {
  const size_t num_tokens = hidden_states->shape[0];
  Tensor topk_weights_tensor(gating_output->location, TYPE_FP32, {num_tokens, expert_topk_}, topk_weights_ptr_);
  Tensor topk_ids_tensor(gating_output->location, TYPE_INT32, {num_tokens, expert_topk_}, topk_ids_ptr_);
  Status status = grouped_topk_layer_->Forward(gating_output, &topk_weights_tensor, &topk_ids_tensor, e_score_bias);
  if (!status.OK()) {
    KLLM_LOG_ERROR << fmt::format("ExecuteGroupedTopk ERROR: failed to forward grouped_topk_layer");
    return status;
  }

  if (enable_dump_eplb_data_) {
    DumpEplbData(topk_ids_tensor);
  }

  if (enable_load_eplb_config_ && eplb_map->shape[0] > static_cast<size_t>(layer_idx_)) {
    size_t num_experts = eplb_map->shape[1];
    void* layer_expert_ptr =
        reinterpret_cast<uint8_t*>(eplb_map->GetPtr<void>()) + layer_idx_ * num_experts * sizeof(int);
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                        reinterpret_cast<int32_t*>(layer_expert_ptr),
                                        context_->device->GetPrimaryComputeStream().Get());
  }

  // TODO(jinxcwu) 支持deepep的量化传输
  if (using_deepep_) {
    // LCOV_EXCL_START
    std::vector<Tensor> deepep_input_tensors = {*hidden_states, topk_ids_tensor, topk_weights_tensor};
    std::vector<Tensor> deepep_output_tensors = {*output, *workspace};
    KLLM_LOG_DEBUG << fmt::format("ExecuteGroupedTopk: Dispatch shape {} {}", deepep_input_tensors[0].shape[0],
                                  deepep_input_tensors[1].shape[0]);

    if (use_low_latency_mode_) {
      deepep_output_tensors.emplace_back(gating_output->location, TYPE_INT32, std::vector<size_t>{expert_num_per_node_},
                                         recv_expert_count_ptr_);
    }

    Dispatch(deepep_input_tensors, deepep_output_tensors);

    if (!use_low_latency_mode_) {
      topk_ids_tensor.shape[0] = deepep_output_tensors[0].shape[0];
      topk_weights_tensor.shape[0] = deepep_output_tensors[0].shape[0];
    }
    workspace->shape = deepep_output_tensors[1].shape;
    // LCOV_EXCL_STOP
  }
  return Status();
}

Status CutlassMoeLayer::Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return expert_parallel_manager_->Dispatch(input_tensors, output_tensors, context_->device->GetPrimaryComputeStream());
}

Status CutlassMoeLayer::Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return expert_parallel_manager_->Combine(input_tensors, output_tensors, context_->device->GetPrimaryComputeStream());
}

Status CutlassMoeLayer::DumpEplbData(Tensor& topk_ids) {
  topk_ids.SaveToNpyFile(
      fmt::format("{}/layer_{}/topk_ids_{}_{}.npy", eplb_dump_path_, layer_idx_, eplb_dump_step_, rank_));
  eplb_dump_step_ += 1;
  return Status();
}

void CutlassMoeLayer::AdaptLowLatencyDispatchOutput(const int* recv_expert_count, int num_experts_per_node,
                                                    int max_tokens_per_expert, Stream& stream) {
  // 计算总的 token 数量
  const int total_tokens = num_experts_per_node * max_tokens_per_expert;

  // 计算全局的 num_slots (所有 EP rank 的专家总数)
  const int num_slots = expert_num_per_node_ * global_expert_para_size_;

  KLLM_LOG_DEBUG << fmt::format(
      "Rank[{}] AdaptLowLatencyDispatchOutput: num_experts={}, max_tokens_per_expert={}, "
      "total_tokens={}, ep_rank={}, num_slots={}",
      rank_, num_experts_per_node, max_tokens_per_expert, total_tokens, global_expert_para_rank_, num_slots);

  // 调用 kernel 生成 adapted topk_ids 和 topk_weights
  llm_kernels::nvidia::InvokeAdaptLowLatencyDispatch(
      recv_expert_count, static_cast<int32_t*>(adapted_topk_ids_ptr_), static_cast<float*>(adapted_topk_weights_ptr_),
      num_experts_per_node, max_tokens_per_expert, global_expert_para_rank_, num_slots, stream.Get());

  KLLM_LOG_DEBUG << fmt::format("Rank[{}] AdaptLowLatencyDispatchOutput completed", rank_);
}

Status CutlassMoeLayer::Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1,
                                const Tensor* w2, const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                                const Tensor* e_score_bias, const Tensor* input_ids, const LayerExecContext& exec_ctx) {
  if (set_workspace_buffer_info_) {
    set_workspace_buffer_info_ = false;
    topk_weights_ptr_ = workspace_buffer_->GetPtr<void>();
    topk_ids_ptr_ = reinterpret_cast<uint8_t*>(topk_weights_ptr_) + topk_weights_ptr_size;

    if (use_low_latency_mode_) {
      adapted_topk_weights_ptr_ = reinterpret_cast<uint8_t*>(topk_ids_ptr_) + topk_ids_ptr_size;
      adapted_topk_ids_ptr_ = reinterpret_cast<uint8_t*>(adapted_topk_weights_ptr_) + adapted_topk_weights_size_;
      recv_expert_count_ptr_ = reinterpret_cast<uint8_t*>(adapted_topk_ids_ptr_) + adapted_topk_ids_size_;
      kernel_workspace_ptr_ = reinterpret_cast<uint8_t*>(recv_expert_count_ptr_) + recv_expert_count_size_;

      KLLM_LOG_DEBUG << fmt::format(
          "Rank[{}] LowLatency workspace layout: topk_weights@{}, topk_ids@{}, "
          "adapted_weights@{}, adapted_ids@{}, recv_count@{}, kernel@{}",
          rank_, topk_weights_ptr_, topk_ids_ptr_, adapted_topk_weights_ptr_, adapted_topk_ids_ptr_,
          recv_expert_count_ptr_, kernel_workspace_ptr_);
    } else {
      kernel_workspace_ptr_ = reinterpret_cast<uint8_t*>(topk_ids_ptr_) + topk_ids_ptr_size;
    }

    cutlass_moe_wrapper_->SetWorkspacePtr(kernel_workspace_ptr_);
    if (swiglu_limit_ > 0.0f) {
      swiglu_limit_ptr_ = reinterpret_cast<uint8_t*>(kernel_workspace_ptr_) + kernel_workspace_size;
    }
  }

  if (swiglu_limit_ > 0.0f) {
    cudaMemcpyAsync(swiglu_limit_ptr_, swiglu_limit_host_.data(), expert_num_per_node_ * sizeof(float),
                    cudaMemcpyHostToDevice, context_->device->GetPrimaryComputeStream().Get());
  }

  KTensor fc1_expert_weights_ktensor = TensorToKTensor(*w1);
  KTensor fc2_expert_weights_ktensor = TensorToKTensor(*w2);
  KTensor fc1_weight_scales_ktensor = TensorToKTensor(*(w1->scales));
  KTensor fc2_weight_scales_ktensor = TensorToKTensor(*(w2->scales));
  KTensor fc1_act_scales_ktensor = TensorToKTensor(*(w1->input_scales));
  KTensor fc2_act_scales_ktensor = TensorToKTensor(*(w2->input_scales));
  KTensor fc1_weight_zeros_ktensor;
  KTensor fc2_weight_zeros_ktensor;
  KTensor fc1_alpha_ktensor = TensorToKTensor(*(w1->input_alpha));
  KTensor fc2_alpha_ktensor = TensorToKTensor(*(w2->input_alpha));

  STATUS_CHECK_RETURN(ExecuteGroupedTopk(hidden_states, gating_output, eplb_map, output, workspace, e_score_bias));

  auto AutoInvoke = [&](auto& input_tensor, auto& output_tensor, int num_tokens) {
    ProcessChunks(input_tensor, output_tensor, num_tokens, fc1_expert_weights_ktensor, fc2_expert_weights_ktensor,
                  {fc1_weight_scales_ktensor, fc2_weight_scales_ktensor, fc1_act_scales_ktensor, fc2_act_scales_ktensor,
                   fc1_weight_zeros_ktensor, fc2_weight_zeros_ktensor, fc1_alpha_ktensor, fc2_alpha_ktensor});
  };

  if (using_deepep_) {
    // LCOV_EXCL_START
    size_t dispatch_num_tokens;
    if (use_low_latency_mode_) {
      dispatch_num_tokens = workspace->shape[1];
    } else {
      dispatch_num_tokens = workspace->shape[0];
    }

    if (dispatch_num_tokens > 0) {
      if (use_low_latency_mode_) {
        const size_t flat_token_num = expert_num_per_node_ * dispatch_num_tokens;
        KLLM_LOG_DEBUG << fmt::format(
            "Rank[{}] LowLatency mode: dispatch_num_tokens(per_expert)={}, flat_token_num={}, "
            "expert_num_per_node={}, calling AdaptLowLatencyDispatchOutput",
            rank_, dispatch_num_tokens, flat_token_num, expert_num_per_node_);

        AdaptLowLatencyDispatchOutput(static_cast<const int*>(recv_expert_count_ptr_), expert_num_per_node_,
                                      dispatch_num_tokens, context_->device->GetPrimaryComputeStream());

        workspace->shape = {flat_token_num, expert_hidden_size_};
        AutoInvoke(*workspace, *workspace, flat_token_num);
      } else {
        expert_map_->InvokeExpertMapInverseInplace(static_cast<int32_t*>(topk_ids_ptr_),
                                                   dispatch_num_tokens * expert_topk_,
                                                   context_->device->GetPrimaryComputeStream().Get());
        AutoInvoke(*workspace, *workspace, dispatch_num_tokens);
      }
    }

    if (use_low_latency_mode_) {
      workspace->shape = {expert_num_per_node_, dispatch_num_tokens, expert_hidden_size_};
      KLLM_LOG_DEBUG << fmt::format("Rank[{}] CutlassMoeLayer: LL mode Combine input shape: [{}, {}, {}]", rank_,
                                    expert_num_per_node_, dispatch_num_tokens, expert_hidden_size_);
    }

    std::vector<Tensor> combine_inputs = {*workspace};
    std::vector<Tensor> combine_outputs = {*output, *workspace};
    Combine(combine_inputs, combine_outputs);
    *output = combine_outputs[0];
    // LCOV_EXCL_STOP
  } else {
    output->shape = hidden_states->shape;
    output->dtype = hidden_states->dtype;
    const size_t num_tokens = hidden_states->shape[0];
    AutoInvoke(*hidden_states, *output, num_tokens);
  }
  output->shape = hidden_states->shape;
  output->dtype = hidden_states->dtype;
  return Status();
}

}  // namespace ksana_llm
