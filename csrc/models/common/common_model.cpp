/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/common/common_model.h"

#include <memory>
#include <vector>

#include "device/common_device.h"
#include "device/device_types.h"
#include "fmt/core.h"
#include "profiler/profile_event.h"
#include "profiler/sched_event_tracer.h"
#include "runtime/infer_stage.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/request.h"
#include "utils/singleton.h"
#include "utils/string_utils.h"

#ifdef ENABLE_CUDA
#  include "layers/nvidia/replace_last_token.h"
#endif

namespace ksana_llm {

void RecordRequestSchedEventWithFContext(ForwardingContext& forwarding_context, const char* type,
                                         RequestEventPhase phase) {
  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, type, phase);
}

CommonModel::CommonModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                         std::shared_ptr<Context> context) {
  model_config_ = model_config;
  runtime_config_ = runtime_config;
  context_ = context;
  rank_ = rank;
  GetBufferManager()->SetRank(rank_);

  KLLM_LOG_DEBUG << "Working mode info, is_standalone:" << context_->global->IsStandalone()
                 << ", is_distributed_master:" << context_->global->IsDistributedMaster();
}

CommonModel::~CommonModel() {}

void CommonModel::InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight,
                                std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  prefix_caching_enabled_ = runtime_config_.enable_prefix_caching;
  speculative_decoding_enabled_ = runtime_config_.enable_speculative_decoding;

  size_t free_device_mem_before_init, free_device_mem_after_init, total_device_mem;
  MemGetInfo(&free_device_mem_before_init, &total_device_mem);

  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config_);

  Singleton<Environment>::GetInstance()->GetExpertParallelConfig(expert_parallel_config_);
  model_run_config_ = model_run_config;

  // Init expert_local_rank
  expert_parallel_config_.local_expert_rank = rank_;
  Singleton<Environment>::GetInstance()->SetExpertParallelConfig(expert_parallel_config_);

  model_buffers_.Init(context_, rank_, model_config_, runtime_config_, GetBufferManager());

  {
    // Initialize all forwarding contexts in the buffer
    forwarding_context_ = std::make_unique<ForwardingContext>();
    forwarding_context_->Init(context_, rank_, model_config_, runtime_config_, pipeline_config_,
                              model_buffers_.buffers_.get(), GetBufferManager(), block_cache_layout);
    KLLM_LOG_DEBUG << "Initialized forwarding context buffer.";

    // Allow model subclasses to inject a model-specific AttentionMetaCreator.
    auto custom_creator = CreateCustomAttentionMetaCreator(forwarding_context_->GetModelInput()->GetCreatorConfig());
    if (custom_creator) {
      forwarding_context_->GetModelInput()->ReplaceAttentionMetaCreator(std::move(custom_creator));
    }
  }

  int layer_num_on_node = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
  if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(model_config_.num_layer)) {
    layer_num_on_node += pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
  }

  const int hidden_units = model_config_.size_per_head * model_config_.head_num;

  BlockManagerConfig block_manager_config;
  STATUS_CHECK_FAILURE(Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config));

  // Initialize instances for each layer.
  layer_creation_context_.Init(base_weight, context_, rank_, pipeline_config_, model_config_, runtime_config_,
                               GetBufferManager());

  emb_lookup_layer_ = std::make_shared<EmbLookupLayer>();
  if (model_run_config_.position_encoding == PositionEncoding::LEARNED_ABSOLUTE) {
    Tensor position_weight = base_weight->GetModelWeights("model.embed_positions.weight");
    emb_lookup_layer_->Init(
        {model_run_config_.use_emb_scale, model_run_config_.emb_scale, position_weight.GetPtr<void>()}, runtime_config_,
        context_, rank_);
  } else {
    emb_lookup_layer_->Init({model_run_config_.use_emb_scale, model_run_config_.emb_scale}, runtime_config_, context_,
                            rank_);
  }

  cpu_emb_lookup_layer_ = std::make_shared<CpuEmbLookupLayer>();
  cpu_emb_lookup_layer_->Init({}, runtime_config_, context_, rank_);

  assemble_tokens_hidden_layer_ = std::make_shared<AssembleTokensHiddenLayer>();
  assemble_tokens_hidden_layer_->Init({}, runtime_config_, context_, rank_);

  cast_layer_ = std::make_shared<CastLayer>();
  cast_layer_->Init({}, runtime_config_, context_, rank_);

  input_refit_layer_ = std::make_shared<InputRefitLayer>();
  input_refit_layer_->Init({}, runtime_config_, context_, rank_);

  if (runtime_config_.embed_tokens_use_cpu) {
    DataType input_data_type = TYPE_INT32;
    size_t max_token_num = runtime_config_.max_step_token_num;
    cpu_input_tokens_tensor_ = Tensor(MemoryLocation::LOCATION_HOST, input_data_type, {max_token_num});
    cpu_tokens_emb_tensor_ = Tensor(MemoryLocation::LOCATION_HOST, input_data_type, {max_token_num * hidden_units});
  }

  KLLM_LOG_DEBUG << "Total buffer tensors memory used: " << (GetBufferTensorsMemoryUsed() >> 20) << " MB";

  ModelCreationConfig model_creation_config;
  model_creation_config.layernorm_config.layernorm_eps = model_config_.layernorm_eps;
  model_creation_config.layernorm_config.activation_function = model_config_.activation_function;

  // Flash Attention requires the input shape to match the actual token length.
  // When dealing with prefix_cache or speculative decoding, it is necessary to
  // first fill in the missing parts
  // NOTE(jinxcwu) 只有qwen2/2.5/3-vl模型使用mrope，有mrope_section字段的内容
  if (!model_config_.rope_scaling_factor_config.mrope_section.empty()) {
    mrotary_section_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {3});
    MemcpyAsync(mrotary_section_tensor_.GetPtr<void>(), model_config_.rope_scaling_factor_config.mrope_section.data(),
                3 * sizeof(int), MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  }
  if (model_config_.type == "arc_hunyuan_video") {
    xdrotary_section_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {4});
    MemcpyAsync(xdrotary_section_tensor_.GetPtr<void>(), model_config_.rope_scaling_factor_config.xdrope_section.data(),
                4 * sizeof(int), MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  }
  bool reuse_prefix_config = prefix_caching_enabled_ || speculative_decoding_enabled_;
  model_creation_config.Init(model_config_, runtime_config_, model_buffers_.cos_sin_cache_tensor_,
                             model_buffers_.compress_cos_sin_cache_tensor_, model_run_config_.position_encoding,
                             reuse_prefix_config, layer_num_on_node, mrotary_section_tensor_.GetPtr<const int>(false),
                             xdrotary_section_tensor_.GetPtr<const int>(false));

  // create matmul layer
  CreateLayers(layer_creation_context_, model_creation_config);

  if (context_->global->IsSamplingNode()) {
    lm_head_ = std::make_shared<Linear>("lm_head.weight", layer_creation_context_,
                                        model_creation_config.attn_config.model_config.quant_config.backend);
    if (model_run_config_.layernorm_position == LayerNormPosition::PRE_NORM) {
      lm_head_prenorm_ =
          std::make_shared<Layernorm>("model.norm.weight", model_config_.layernorm_eps, layer_creation_context_);
    }
  }

  MemGetInfo(&free_device_mem_after_init, &total_device_mem);
  KLLM_LOG_INFO << "rank=" << rank_ << ": BufferManager used "
                << GetBufferManager()->GetBufferTensorsMemoryUsed() / (1024 * 1024)
                << "MB, total_device_mem=" << total_device_mem / (1024 * 1024)
                << "MB, free_device_mem_before_init=" << free_device_mem_before_init / (1024 * 1024)
                << "MB, free_device_mem_after_init=" << free_device_mem_after_init / (1024 * 1024) << "MB";
}

float* CommonModel::GetLogitsPtr() {
  return forwarding_context_->GetModelOutput()->logits_tensor.template GetPtr<float>(false);
}

void CommonModel::SetInflightResourceManager(InflightResourceManager* mgr) {
  if (forwarding_context_) {
    forwarding_context_->SetInflightResourceManager(mgr);
  }
}

void CommonModel::SetForwardProfileMetrics(ProfileMetrics* pm, ForwardPerfMetrics* current) {
  if (forwarding_context_) {
    forwarding_context_->profile_metrics = pm;
    forwarding_context_->current_forward_metrics = current;
  }
}

Status CommonModel::EmbedTokensUseCpu(Tensor& embedding_weight, std::vector<ForwardRequest>& forward_reqs,
                                      ForwardingContext& forwarding_context) {
  void* input_tokens_ptr = cpu_input_tokens_tensor_.GetPtr<void>();
  memcpy(input_tokens_ptr, forwarding_context.GetModelInput()->input_ids_cpu.data(),
         forwarding_context.GetModelInput()->input_ids_cpu.size() * sizeof(int));
  cpu_input_tokens_tensor_.shape = {forwarding_context.GetModelInput()->input_ids_cpu.size()};

  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
  auto* typed = static_cast<CpuEmbLookupLayer*>(cpu_emb_lookup_layer_.get());
  typed->Forward(&cpu_input_tokens_tensor_, &cpu_tokens_emb_tensor_, &embedding_weight, &residual_buffer[0]);
  return Status();
}

Status CommonModel::EmbedTokensUseGpu(Tensor& embedding_weight, ForwardingContext& forwarding_context) {
  // Wait the computation of input_ids.
  StreamWaitEvent(context_->device->GetPrimaryComputeStream(), forwarding_context.GetModelInput()->input_ids_event);
  if (model_run_config_.emb_lookup_use_rotary_embedding_pos) {
    StreamWaitEvent(context_->device->GetPrimaryComputeStream(),
                    forwarding_context.GetModelInput()->rotary_embedding_event);
  }

  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);

  KLLM_LOG_DEBUG << "CommonModel::EmbedTokensUseGpu before emb residual_buffer.shape:"
                 << Vector2Str(residual_buffer[0].shape);
  auto* typed_emb = static_cast<EmbLookupLayer*>(emb_lookup_layer_.get());
  if (model_run_config_.emb_lookup_use_rotary_embedding_pos) {
    STATUS_CHECK_RETURN(typed_emb->Forward(
        &forwarding_context.GetModelInput()->input_ids, &forwarding_context.GetModelInput()->input_offset_uint64_tensor,
        &forwarding_context.GetModelInput()->input_prefix_uint64_tensor, &embedding_weight, &residual_buffer[0],
        &forwarding_context.GetModelInput()->flash_meta->rotary_embedding_pos));
    KLLM_LOG_DEBUG << "CommonModel::EmbedTokensUseGpu after emb residual_buffer.shape:"
                   << Vector2Str(residual_buffer[0].shape);
  } else {
    STATUS_CHECK_RETURN(typed_emb->Forward(
        &forwarding_context.GetModelInput()->input_ids, &forwarding_context.GetModelInput()->input_offset_uint64_tensor,
        &forwarding_context.GetModelInput()->input_prefix_uint64_tensor, &embedding_weight, &residual_buffer[0]));
    KLLM_LOG_DEBUG << "CommonModel::EmbedTokensUseGpu after emb residual_buffer.shape:"
                   << Vector2Str(residual_buffer[0].shape);
  }

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->global->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(forwarding_context.GetModelOutput()->compute_ready_event, context_->device->GetPrimaryComputeStream());
    StreamWaitEvent(context_->device->GetCommStream(), forwarding_context.GetModelOutput()->compute_ready_event);
  }

  if (forwarding_context.GetModelCommunicator()) {
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
    forwarding_context.GetModelCommunicator()->AllGather({residual_buffer[0], hidden_buffer_tensors_1[0]},
                                                         residual_buffer);
  }
  return Status();
}

bool CommonModel::UpdateResponse(std::vector<ForwardRequest>& forward_reqs, Tensor& output, const std::string& stage) {
  bool ret = true;
  int req_offset = 0;
  for (ForwardRequest& req : forward_reqs) {
    int output_token_num = req.forwarding_tokens->size();
    if (!req.request_target) {
      ret = false;
      continue;
    }
    const auto it = req.request_target->find(stage);
    if (it == req.request_target->end()) {
      ret = false;
      continue;
    } else if (it->second.token_reduce_mode != TokenReduceMode::GATHER_ALL) {
      // GATHER_TOKEN_ID for "logits"
      ret = false;
      continue;
    }
    // Determine whether to exit early
    ret &= req.request_target->size() == req.response->size();
    if (rank_ != 0) continue;
    int output_len = 0;
    std::vector<std::pair<int, int>> slice_pos = it->second.slice_pos;
    // If specific token IDs are provided, add their positions to slice_pos.
    if (it->second.token_id.size() != 0) {
      std::set<int> token_id_set(it->second.token_id.begin(), it->second.token_id.end());
      for (int i = 0; i < output_token_num; i++) {
        if (token_id_set.count(req.forwarding_tokens->at(i)) > 0) {
          slice_pos.push_back({i, i});
        }
      }
    }
    // Calculate the total output length based on slice positions.
    for (auto [l, r] : slice_pos) {
      output_len += r - l + 1;
    }
    // Calculate the size of each chunk based on the output tensor's data type and shape.
    size_t chunk_size = GetTypeSize(output.dtype) * output.shape[1];
    // Update the response tensor with the sliced data.
    PythonTensor& ret_tensor = (*req.response)[stage];
    ret_tensor.shape = {static_cast<size_t>(output_len), output.shape[1]};
    ret_tensor.dtype = GetTypeString(output.dtype);
    ret_tensor.data.resize(output_len * chunk_size);
    if (stage == "logits") {
      // Update slice_pos as {[0, output_len - 1]} to skip cutting.
      slice_pos = {{0, output_len - 1}};
      output_token_num = output_len;
    }
    req_offset += output_token_num;
    output_len = 0;
    // Copy data from the output tensor to the output_data buffer based on slice positions.
    for (auto [l, r] : slice_pos) {
      MemcpyAsync(ret_tensor.data.data() + output_len * chunk_size,
                  output.GetPtr<void>() + (req_offset - output_token_num + l) * chunk_size, (r - l + 1) * chunk_size,
                  MEMCPY_DEVICE_TO_HOST, context_->device->GetPrimaryComputeStream());
      output_len += r - l + 1;
    }
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }
  return ret;
}

std::vector<Tensor>& CommonModel::GetHiddenUnitBufferTensors(ForwardingContext& forwarding_context) {
#ifdef ENABLE_ACL
  if (forwarding_context.GetModelInput()->infer_stage == InferStage::kContext) {
    return distributed_device_buffer_prefill_;
  } else {
#endif
    return distributed_device_buffer_;
#ifdef ENABLE_ACL
  }
#endif
}

void CommonModel::SetHiddenUnitBufferTensors(HiddenUnitDeviceBuffer* const hidden_unit) {
#ifdef ENABLE_ACL
  if (forwarding_context.GetModelInput()->infer_stage == InferStage::kContext) {
    distributed_device_buffer_prefill_ = {hidden_unit->prefill_tensor};
  } else {
#endif
    distributed_device_buffer_ = {hidden_unit->tensor};
#ifdef ENABLE_ACL
  }
#endif
}

Status CommonModel::PrepareModelInputGpu(std::vector<ForwardRequest>& forward_reqs, RunMode run_mode,
                                         size_t model_input_slot) {
  forwarding_context_->PrepareModelInputGpu(forward_reqs, run_mode, model_input_slot);
  return Status();
}

void CommonModel::GetCurBatchPairs(size_t model_input_slot, const int64_t** out_pairs_dev, int* out_batch_size) {
  if (out_pairs_dev) {
    *out_pairs_dev = nullptr;
  }
  if (out_batch_size) {
    *out_batch_size = 0;
  }
  if (forwarding_context_ == nullptr || model_input_slot > 1) {
    return;
  }
  auto& mi = forwarding_context_->GetModelInput(model_input_slot);
  if (mi == nullptr || mi->cur_batch_pairs_tensor.shape.empty()) {
    return;
  }
  const int64_t total_elems = static_cast<int64_t>(mi->cur_batch_pairs_tensor.shape[0]);
  if (total_elems <= 0 || (total_elems & 1) != 0) {
    return;
  }
  if (out_pairs_dev) {
    *out_pairs_dev = mi->cur_batch_pairs_tensor.GetPtr<int64_t>();
  }
  if (out_batch_size) {
    *out_batch_size = static_cast<int>(total_elems / 2);
  }
}

Status CommonModel::Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                            std::vector<ForwardRequest>& forward_reqs, HiddenUnitDeviceBuffer* hidden_unit,
                            RunMode run_mode, size_t model_input_slot, const int64_t* prev_ring_dev,
                            bool gpu_input_already_prepared) {
  // Serialize against concurrent Forward() invocations on the same CommonModel instance.
  // Required on the sampling rank where HandleExecute (main, RunMode::kMain) and
  // HandlePostprocess -> MtpForward (RunMode::kNextN) can otherwise overlap under
  // max_pp_batch_num >= 2 and race on the shared forwarding_context_ state
  // (ModelInput / *_meta host vectors, ForwardingBuffers TensorBuffer in-use flags).
  // Released on function return, after all device kernels have been launched; H2D copies from the
  // host vectors are issued synchronously w.r.t. the host (pageable memory is staged inside
  // cudaMemcpyAsync before return), so the lock does not need to span StreamSynchronize.
  std::lock_guard<std::mutex> forward_lock(forward_mutex_);

  SetHiddenUnitBufferTensors(hidden_unit);

  time_t start_time_ms = ProfileTimer::GetCurrentTimeInMs();
  KLLM_LOG_DEBUG << "start forward, rank=" << rank_;

  PROFILE_EVENT_SCOPE(CommonModel_Forward,
                      fmt::format("CommonModel_Forward_rank{}", forwarding_context_->GetCurrentRank()),
                      forwarding_context_->GetCurrentRank());

  forwarding_context_->GetBatchRequestSchedInfo() = BuildBatchRequestSchedInfoFromForwardingReqs(forward_reqs);
  forwarding_context_->UpdateBeforeForward(forward_reqs, run_mode, model_input_slot, gpu_input_already_prepared);

  if (!context_->global->IsDistributedWorker()) {
    RecordRequestSchedEventWithFContext(*forwarding_context_, "PrepareForwarding", RequestEventPhase::End);
  }

#ifdef ENABLE_CUDA
  // build-ahead: prev_ring_dev 非空时在 Embed 前 launch ReplaceLastTokenKernel 回填 placeholder。
  if (prev_ring_dev != nullptr) {
    auto& mi = forwarding_context_->GetModelInput();
    KLLM_CHECK_WITH_INFO(mi != nullptr, "prev_ring_dev != nullptr but ModelInput is null in CommonModel::Forward.");
    KLLM_CHECK_WITH_INFO(!forward_reqs.empty(),
                         "prev_ring_dev != nullptr but forward_reqs is empty in CommonModel::Forward.");
    StreamWaitEvent(context_->device->GetPrimaryComputeStream(), mi->input_ids_event);
    PROFILE_EVENT_SCOPE(ReplaceLastToken_, fmt::format("ReplaceLastToken_slot{}", model_input_slot),
                        forwarding_context_->GetCurrentRank());
    LaunchReplaceLastTokenKernel(mi->input_ids.GetPtr<int32_t>(), mi->cur_batch_pairs_tensor.GetPtr<int64_t>(),
                                 prev_ring_dev, static_cast<int>(forward_reqs.size()),
                                 context_->device->GetPrimaryComputeStream().Get());
  }
#endif

  if (!context_->global->IsDistributedWorker() || (run_mode == RunMode::kNextN && context_->global->IsSamplingNode())) {
    RecordRequestSchedEventWithFContext(*forwarding_context_, "EmbLookup", RequestEventPhase::Begin);
    LookupEmbedding(*forwarding_context_, base_weight, forward_reqs, run_mode);
    RecordRequestSchedEventWithFContext(*forwarding_context_, "EmbLookup", RequestEventPhase::End);
  }

  forwarding_context_->SetIsForwardingLayers(true);
  if (!runtime_config_.is_profile_mode) {
    LayerForward(*forwarding_context_, run_mode);
  } else {
    // Used for layer forwarding performance profile
    for (size_t idx = 0; idx < g_profile_layer_forwarding_round; idx++) {
      LayerForward(*forwarding_context_, run_mode);
    }
  }
  forwarding_context_->SetIsForwardingLayers(false);

  if (context_->global->IsSamplingNode()) {
    LmHead(*forwarding_context_, base_weight, forward_reqs, run_mode);
  }

  time_t end_time_ms = ProfileTimer::GetCurrentTimeInMs();
  KLLM_LOG_DEBUG << "CommonModel Forward, time cost=" << end_time_ms - start_time_ms << "ms";
  return Status();
}

Status CommonModel::LookupEmbedding(ForwardingContext& forwarding_context,
                                    std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                    std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode) {
  KLLM_LOG_DEBUG << "start lookup embedding, rank=" << rank_ << "";
  PROFILE_EVENT_SCOPE(CommonModel_LookupEmbedding, "CommonModel_LookupEmbedding", forwarding_context.GetCurrentRank());
  // CPU embedding lookup
  // The output is stored in `residual_buffer` for residual connection in common
  // decoder.
  Tensor embedding_weight = base_weight->GetModelWeights("model.embed_tokens.weight");
  if (embedding_weight.location == MemoryLocation::LOCATION_HOST) {
    EmbedTokensUseCpu(embedding_weight, forward_reqs, forwarding_context);
  }

  // GPU embedding lookup
  // The output is stored in `residual_buffer` for residual connection in common
  // decoder.
  if (embedding_weight.location == MemoryLocation::LOCATION_DEVICE) {
    EmbedTokensUseGpu(embedding_weight, forwarding_context);
  }

  // refit input needs to be processed only in the multi-token forwarding.
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  if (is_multi_token_forward && run_mode == RunMode::kMain) {
    std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
    std::vector<Tensor>& emb_tensors = forwarding_context.GetModelInput()->cpu_input_refit_tensor.emb_tensors;
    const Tensor& pos_tensor = forwarding_context.GetModelInput()->cpu_input_refit_tensor.pos_tensor;
    std::vector<const Tensor*> emb_ptrs;
    emb_ptrs.reserve(emb_tensors.size());
    for (const Tensor& t : emb_tensors) emb_ptrs.push_back(&t);
    input_refit_layer_->Forward(emb_ptrs, &pos_tensor, &residual_buffer[0]);
  }

  if (initial_mhc_ && run_mode != RunMode::kNextN) {
    // Expand residual_buffer from [t, d] to [t, hc, d]
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
    std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
    MemcpyAsync(hidden_buffer_tensors_1[0].GetPtr<void>(), residual_buffer[0].GetPtr<void>(),
                residual_buffer[0].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
                forwarding_context.GetContext()->device->GetPrimaryComputeStream());
    hidden_buffer_tensors_1[0].shape = residual_buffer[0].shape;
    hidden_buffer_tensors_1[0].dtype = residual_buffer[0].dtype;
    initial_mhc_->InitialForward(hidden_buffer_tensors_1, residual_buffer);
  }
  return Status();
}

Status CommonModel::LmHead(ForwardingContext& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                           std::vector<ForwardRequest>& forward_reqs, RunMode run_mode) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
  RecordRequestSchedEventWithFContext(forwarding_context, "LmHead", RequestEventPhase::Begin);

  // save unnormed hidden result if enable MTP model
  if (runtime_config_.mtp_step_num > 0) {
    if (model_run_config_.return_hidden_states_before_norm) {
      auto& mtp_hidden_tensor = forwarding_context.GetForwardingBuffers()->mtp_hidden_buffer_tensors[0];
      mtp_hidden_tensor.shape = residual_buffer[0].shape;
      mtp_hidden_tensor.dtype = residual_buffer[0].dtype;
      MemcpyAsync(mtp_hidden_tensor.template GetPtr<void>(), residual_buffer[0].template GetPtr<void>(),
                  residual_buffer[0].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
                  context_->device->GetPrimaryComputeStream());
    }
  }

  // Assemble early when there is no need to save mtp_hidden and no need for request_target
  const bool assemble_early =
      (runtime_config_.mtp_step_num == 0 || model_run_config_.return_hidden_states_before_norm) &&
      !forwarding_context.GetModelInput()->has_request_target;

  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  if (assemble_early) {
    // assemble last token
    // The input is stored in `residual_buffer`.
#ifdef ENABLE_CUDA
    auto* typed = static_cast<AssembleTokensHiddenLayer*>(assemble_tokens_hidden_layer_.get());
    STATUS_CHECK_RETURN(typed->Forward(&residual_buffer[0],
                                       &forwarding_context.GetModelInput()->logits_idx_uint64_tensor,
                                       &hidden_buffer_tensors_0[0]));
#elif defined(ENABLE_ACL)
    STATUS_CHECK_RETURN(static_cast<AssembleTokensHiddenLayer*>(assemble_tokens_hidden_layer_.get())
                            ->Forward(&residual_buffer[0], &forwarding_context.GetModelInput()->last_token_index_tensor,
                                      &hidden_buffer_tensors_0[0]));
#endif
  }

  auto& input_tensors = assemble_early ? hidden_buffer_tensors_0 : residual_buffer;

  // Squeeze residual_buffer from [t, hc, d] to [t, d]
  if (run_mode == RunMode::kNextN && nextn_final_mhc_) {
    // Use MTP-specific hc_head for NextN mode
    nextn_final_mhc_->FinalForward(input_tensors, input_tensors);
  } else if (final_mhc_) {
    final_mhc_->FinalForward(input_tensors, input_tensors);
  }

  if (is_multi_token_forward && run_mode == RunMode::kMain) {
    if (UpdateResponse(forward_reqs, residual_buffer[0], "transformer")) {
      StreamSynchronize(context_->device->GetPrimaryComputeStream());
      input_refit_layer_->Clear();
      return Status();
    }
  }

  // final norm
  // Only pre norm model performs final norm.
  // Both input and output are in `residual_buffer`.
  if (run_mode == RunMode::kNextN && nextn_shared_head_norm_) {
    // Use MTP-specific shared_head.norm for NextN mode
    nextn_shared_head_norm_->Forward(input_tensors[0], input_tensors[0]);
  } else if (lm_head_prenorm_) {
    lm_head_prenorm_->Forward(input_tensors[0], input_tensors[0]);
  }

  // save normed hidden result if enable MTP model
  if (runtime_config_.mtp_step_num > 0) {
    if (!model_run_config_.return_hidden_states_before_norm) {
      auto& mtp_hidden_tensor = forwarding_context.GetForwardingBuffers()->mtp_hidden_buffer_tensors[0];
      mtp_hidden_tensor.shape = residual_buffer[0].shape;
      mtp_hidden_tensor.dtype = residual_buffer[0].dtype;
      MemcpyAsync(mtp_hidden_tensor.template GetPtr<void>(), residual_buffer[0].template GetPtr<void>(),
                  residual_buffer[0].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
                  context_->device->GetPrimaryComputeStream());
    }
  }

  if (is_multi_token_forward && run_mode == RunMode::kMain) {
    if (UpdateResponse(forward_reqs, residual_buffer[0], "layernorm")) {
      StreamSynchronize(context_->device->GetPrimaryComputeStream());
      input_refit_layer_->Clear();
      return Status();
    }
  }

  if (!assemble_early) {
    // assemble last token
    // The input is stored in `residual_buffer`.
#ifdef ENABLE_CUDA
    auto* typed = static_cast<AssembleTokensHiddenLayer*>(assemble_tokens_hidden_layer_.get());
    STATUS_CHECK_RETURN(typed->Forward(&residual_buffer[0],
                                       &forwarding_context.GetModelInput()->logits_idx_uint64_tensor,
                                       &hidden_buffer_tensors_0[0]));
#elif defined(ENABLE_ACL)
    STATUS_CHECK_RETURN(static_cast<AssembleTokensHiddenLayer*>(assemble_tokens_hidden_layer_.get())
                            ->Forward(&residual_buffer[0], &forwarding_context.GetModelInput()->last_token_index_tensor,
                                      &hidden_buffer_tensors_0[0]));
#endif
  }

  // lm_head
  PROFILE_EVENT_SCOPE(CommonModel_LmHead_, "CommonModel_LmHead", forwarding_context.GetCurrentRank());

  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
  if (forwarding_context.GetModelCommunicator() && runtime_config_.enable_full_shared_expert) {
    forwarding_context.GetModelCommunicator()->ReduceSum(hidden_buffer_tensors_0, hidden_buffer_tensors_1,
                                                         is_multi_token_forward, /*use_custom*/ false);
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }

  STATUS_CHECK_RETURN(lm_head_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0]));
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->global->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(forwarding_context.GetModelOutput()->compute_ready_event, context_->device->GetPrimaryComputeStream());
    StreamWaitEvent(context_->device->GetCommStream(), forwarding_context.GetModelOutput()->compute_ready_event);
  }

  if (forwarding_context.GetModelCommunicator()) {
    forwarding_context.GetModelCommunicator()->AllGather({hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0]},
                                                         hidden_buffer_tensors_0);
  }

  if (is_multi_token_forward && run_mode == RunMode::kMain) {
    if (UpdateResponse(forward_reqs, hidden_buffer_tensors_0[0], "logits")) {
      StreamSynchronize(context_->device->GetPrimaryComputeStream());
      return Status();
    }
  }

  PROFILE_EVENT_SCOPE(CommonModel_Cast_, fmt::format("CommonModel_Cast"), forwarding_context.GetCurrentRank());
  forwarding_context.UpdateAfterForward(forward_reqs);
  std::vector<Tensor> logits_buffer{forwarding_context.GetModelOutput()->logits_tensor};
  const auto& fs = forwarding_context.GetAttentionForwardContext().forward_shape;
  auto* typed_cast_layer = static_cast<CastLayer*>(cast_layer_.get());
  STATUS_CHECK_RETURN(
      typed_cast_layer->Forward(&hidden_buffer_tensors_0[0], fs.shape[0], fs.shape[1], fs.shape[2], &logits_buffer[0]));

#ifndef ENABLE_CUDA
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
#endif
  RecordRequestSchedEventWithFContext(forwarding_context, "LmHead", RequestEventPhase::End);
  input_refit_layer_->Clear();
  return Status();
}

}  // namespace ksana_llm
