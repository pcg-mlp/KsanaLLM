/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/forwarding_context.h"

#include "runtime/fast_path_controller.h"
#include "utils/singleton.h"

namespace ksana_llm {

size_t ComputeDecodeOnlyStepTokenNum(size_t max_batch_size, size_t max_decode_tokens_per_req,
                                     size_t attn_data_parallel_size) {
  // max_decode_tokens_per_req 已综合 MTP / PTP / generated token，公式与 schedule_config_parser:
  //   max_decode_tokens_per_req = mtp_step_num + 2 * ptp_step_num + 1
  // 一致；至少 1（保护配置异常时 buffer 不为 0）
  const size_t effective = max_decode_tokens_per_req > 0 ? max_decode_tokens_per_req : 1;
  return max_batch_size * effective * attn_data_parallel_size;
}

size_t SelectEffectiveStepTokenNum(const RuntimeConfig& runtime_config,
                                   const BatchSchedulerConfig& batch_scheduler_config) {
  if (runtime_config.is_decode_only) {
    return ComputeDecodeOnlyStepTokenNum(runtime_config.max_batch_size,
                                         batch_scheduler_config.max_decode_tokens_per_req,
                                         runtime_config.parallel_basic_config.attn_data_parallel_size);
  }
  return runtime_config.max_step_token_num;
}

std::string FormatDecodeOnlyShrinkLog(const RuntimeConfig& runtime_config,
                                      const BatchSchedulerConfig& batch_scheduler_config, size_t effective_token_num) {
  return fmt::format(
      "ForwardingBuffers::Init decode-only: shrink token_num {} -> {} "
      "(max_batch_size={}, max_decode_tokens_per_req={} [mtp={}, ptp={}], attn_dp={})",
      runtime_config.max_step_token_num, effective_token_num, runtime_config.max_batch_size,
      batch_scheduler_config.max_decode_tokens_per_req, batch_scheduler_config.mtp_step_num,
      batch_scheduler_config.ptp_step_num, runtime_config.parallel_basic_config.attn_data_parallel_size);
}

void ForwardingBuffers::CalculateBuffersShape(std::shared_ptr<Context> context, const size_t batch_size,
                                              const size_t max_token_num, const DataType& weight_type) {
  auto env = Singleton<Environment>::GetInstance();
  const size_t tensor_para_size = runtime_config.parallel_basic_config.tensor_parallel_size;
  const size_t head_num = model_config.head_num;
  const size_t size_per_head = model_config.size_per_head;
  const size_t hidden_units = model_config.hidden_units;
  const size_t head_num_per_tp = head_num / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  const size_t num_kv_heads_per_tp =
      DivRoundUp(model_config.num_key_value_heads, runtime_config.parallel_basic_config.attn_tensor_parallel_size);
  size_t vocab_size_pad = DivRoundUp(model_config.vocab_size, tensor_para_size) * tensor_para_size;

  BatchSchedulerConfig batch_scheduler_config;
  env->GetBatchSchedulerConfig(batch_scheduler_config);
  const size_t max_logits_tokens = batch_size * batch_scheduler_config.max_decode_tokens_per_req;
  const bool enable_mtp = batch_scheduler_config.mtp_step_num > 0;

  size_t inter_size_per_tp = model_config.inter_size;
  if (!runtime_config.enable_full_shared_expert) {
    inter_size_per_tp /= tensor_para_size;
  }
  if (model_config.has_shared_experts) {
    size_t shared = model_config.moe_config.shared_expert_inter_size;
    if (!runtime_config.enable_full_shared_expert) {
      // When enable_full_shared_expert is enabled, each GPU stores the complete shared experts without tensor
      // parallelism sharding across devices.
      shared /= tensor_para_size;
    }
    inter_size_per_tp = std::max(inter_size_per_tp, shared);
  }
  KLLM_LOG_DEBUG << fmt::format("inter_size_per_tp = {}", inter_size_per_tp);

  // inter_size_per_tp * 2 is used for the output of the fused gate_proj and up_proj in mlp
  const size_t qkv_head_num = model_config.use_mla ? head_num_per_tp : head_num_per_tp + 2 * num_kv_heads_per_tp;
  const size_t max_dim =
      std::max({qkv_head_num * size_per_head, hidden_units * model_config.mhc_config.hc_mult, inter_size_per_tp * 2});
  // 2 is for concatenate of embedding_norm and hidden norm in mtp
  const size_t mtp_concat_hidden_size = (enable_mtp && context->global->IsSamplingNode()) ? hidden_units * 2 : 0;
  size_t all_to_all_hidden_size = 0;
  if (runtime_config.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_LOW_LATENCY) {
    // Use deepep_low_latency_max_tokens to limit buffer sizing instead of max_step_token_num.
    // shared_buffer = max_token_num * shared_buffer_unit_size, so we scale all_to_all_hidden_size
    // so that: max_token_num * all_to_all_hidden_size >=
    //   local_num_experts * expert_world_size * ll_tokens * max(hidden, inter*2)
    size_t ll_tokens = runtime_config.deepep_low_latency_max_tokens > 0 ? runtime_config.deepep_low_latency_max_tokens
                                                                        : runtime_config.max_step_token_num;
    ExpertParallelConfig expert_parallel_config;
    env->GetExpertParallelConfig(expert_parallel_config);
    size_t local_num_experts = expert_parallel_config.local_num_experts;
    size_t expert_world_size = runtime_config.parallel_basic_config.expert_world_size;
    all_to_all_hidden_size =
        (local_num_experts * expert_world_size * ll_tokens * std::max(hidden_units, inter_size_per_tp * 2) +
         max_token_num - 1) /
        max_token_num;
  } else if (runtime_config.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_NORMAL) {
    all_to_all_hidden_size =
        runtime_config.parallel_basic_config.expert_world_size * std::max(hidden_units, inter_size_per_tp * 2);
  }
  size_t shared_buffer_unit_size = std::max({inter_size_per_tp, hidden_units * model_config.mhc_config.hc_mult,
                                             mtp_concat_hidden_size, all_to_all_hidden_size});

  size_t mla_hidden_buffer_size = 0;
  if (model_config.use_mla && !model_config.use_dsv4_attn) {
    const size_t qk_nope_head_dim = model_config.mla_config.qk_nope_head_dim;
    const size_t qk_rope_head_dim = model_config.mla_config.qk_rope_head_dim;
    const size_t v_head_dim = model_config.mla_config.v_head_dim;
    const size_t kv_lora_rank = model_config.mla_config.kv_lora_rank;

    // For buffer reuse of MlaFlashAtten, see MlaAttenVarlen for details.
    // max (q, k, v)
    const size_t mla_flash_attn_size = std::max((qk_nope_head_dim + qk_rope_head_dim), (v_head_dim));
    // shared_buffer is also used to store one of q,k,v
    shared_buffer_unit_size = std::max(shared_buffer_unit_size, head_num_per_tp * mla_flash_attn_size);
    size_t mla_max_dim = std::max(max_dim, head_num_per_tp * mla_flash_attn_size);
    if (model_config.use_dsa) {
      // DeepSeek Sparse MLA applies weight absorption to both prefill and decode tokens
      mla_max_dim = std::max(mla_max_dim, head_num_per_tp * (kv_lora_rank + qk_rope_head_dim));
      shared_buffer_unit_size = std::max(shared_buffer_unit_size, head_num_per_tp * kv_lora_rank);
    }
    // For buffer reuse of MlaPageAtten, see MlaPagedAttention for details.
    // [absorb_q_nope][absorb_q_rope][quant_absorb_q_nope][quant_absorb_q_rope][flash_mla_workspace]
    const size_t mla_page_attn_size = head_num_per_tp * (kv_lora_rank + qk_rope_head_dim) * 2;
    vocab_size_pad = std::max(vocab_size_pad, mla_page_attn_size);

    const size_t token_num_per_dp = max_token_num;
    mla_hidden_buffer_size = token_num_per_dp * mla_max_dim;

    KLLM_LOG_INFO << fmt::format(
        "head_num_per_tp = {}, qk_nope_head_dim = {}, qk_rope_head_dim = {}, v_head_dim = {}, kv_lora_rank = {}, "
        "mla_page_attn_size = {}, vocab_size_pad = {}, max_dim = {}, mla_flash_attn_size = {}, mla_max_dim={}, "
        "mla_hidden_buffer_size = {}",
        head_num_per_tp, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, mla_page_attn_size,
        vocab_size_pad, max_dim, mla_flash_attn_size, mla_max_dim, mla_hidden_buffer_size);
  }

  size_t hidden_buffer_size =
      std::max(std::max(max_logits_tokens * vocab_size_pad, max_token_num * max_dim), mla_hidden_buffer_size);
  // `shared_buffer_` is shared by `gated_buffer_`, `reduce_buffer_` and `paged_buffer_`.
  const size_t shared_buffer_size = max_token_num * shared_buffer_unit_size;
  if (runtime_config.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_NORMAL ||
      runtime_config.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_LOW_LATENCY) {
    hidden_buffer_size = std::max(hidden_buffer_size, shared_buffer_size);
  }
  KLLM_LOG_INFO << "max_batch_size=" << batch_size << ", vocab_size_pad=" << vocab_size_pad
                << ", max_token_num=" << max_token_num << ", max_dim=" << max_dim << ", hidden_units=" << hidden_units
                << ", inter_size_per_tp=" << inter_size_per_tp << ", hidden_buffer_size=" << hidden_buffer_size
                << ", shared_buffer_size=" << shared_buffer_size;
  buffers_shape_map = {{"hidden_buffer_0", {hidden_buffer_size}},
                       {"hidden_buffer_1", {hidden_buffer_size}},
                       {"shared_buffer", {shared_buffer_size}}};

  // TODO(robertyuan): This buffer is too large
  if (model_config.use_mla) {
    buffers_shape_map["kv_cache_buffer"] = {0};
  } else {
    const size_t max_seq_len = runtime_config.max_seq_len;  // max seq len for one request
    const size_t block_token_num = runtime_config.attn_backend_config.block_token_num;

    // Whether to use FlashInfer for the decode stage.
    if (runtime_config.attn_backend_config.use_flashinfer_for_decode) {
      // FlashInfer workspace buffer layout (elements, dtype: int32_t):
      // - kv_last_page_len: [batch_size]
      // - kv_indptr:        [batch_size + 1]
      // - qo_indptr:        [batch_size + 1]
      // - kv_indices:       [sum of actual blocks, upper bound: batch_size * max_blocks_per_seq]
      // Total (upper bound): batch_size + (batch_size + 1) + (batch_size + 1) + batch_size * max_blocks_per_seq
      //                    = batch_size * (3 + max_blocks_per_seq) + 2
      const size_t max_blocks_per_seq = (max_seq_len + block_token_num - 1) / block_token_num;
      buffers_shape_map["kv_cache_buffer"] = {batch_size * (3 + max_blocks_per_seq) + 2};
    } else {
      // PagedAttention V2 workspace buffer layout:
      // - exp_sums:   [batch_size, num_heads_per_tp, max_num_partitions] (dtype: float)
      // - max_logits: [batch_size, num_heads_per_tp, max_num_partitions] (dtype: float)
      // - tmp_out:    [batch_size, num_heads_per_tp, max_num_partitions, head_size] (dtype: SCALAR_T)
      // Total = batch_size * num_heads_per_tp * max_num_partitions * (num_heads_per_tp + 2)
      // - Since CreateBuffer only supports a single dtype, unifying the type as float for calculation.
      constexpr size_t kPagedAttentionPartitionSize = 512;
      const size_t max_num_partitions = (max_seq_len + kPagedAttentionPartitionSize - 1) / kPagedAttentionPartitionSize;
      buffers_shape_map["kv_cache_buffer"] = {batch_size, max_num_partitions, head_num_per_tp, size_per_head + 2};
    }
  }

  buffers_shape_map["mtp_hidden_buffer_tensors"] = {
      enable_mtp ? max_token_num * model_config.hidden_units * model_config.mhc_config.hc_mult : 0};
}

void ForwardingBuffers::Init(std::shared_ptr<Context> context, const int rank, const ModelConfig& model_config,
                             const RuntimeConfig& runtime_config, BufferManager* const buffer_mgr) {
  this->model_config = model_config;
  this->runtime_config = runtime_config;
  const DataType weight_type = model_config.weight_data_type;
  // LCOV_EXCL_START - Singleton fetch requires full Environment init, covered by integration.
  BatchSchedulerConfig batch_scheduler_config;
  Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
  // LCOV_EXCL_STOP
  const size_t effective_token_num = SelectEffectiveStepTokenNum(runtime_config, batch_scheduler_config);
  if (runtime_config.is_decode_only) {
    KLLM_LOG_INFO << FormatDecodeOnlyShrinkLog(runtime_config, batch_scheduler_config, effective_token_num);
  }
  CalculateBuffersShape(context, runtime_config.max_batch_size, effective_token_num, weight_type);

  Stream* const stream = &(context->device->GetMemoryManageStream());

  // NOTE(karlluo): all create tensor used dynamic memory pool
  hidden_buffer_0 = buffer_mgr->CreateBufferTensor("hidden_buffer_0", buffers_shape_map["hidden_buffer_0"], weight_type,
                                                   MemoryLocation::LOCATION_DEVICE, stream);
  hidden_buffer_1 = buffer_mgr->CreateBufferTensor("hidden_buffer_1", buffers_shape_map["hidden_buffer_1"], weight_type,
                                                   MemoryLocation::LOCATION_DEVICE, stream);
  // When using multicast, place shared_buffer (used for allreduce) at the multicast address
  shared_buffer = buffer_mgr->CreateBufferTensor("shared_buffer", buffers_shape_map["shared_buffer"], weight_type,
                                                 context->device->ext->IsMulticastSupported()
                                                     ? MemoryLocation::LOCATION_MULTICAST
                                                     : MemoryLocation::LOCATION_DEVICE_IPC);
  kv_cache_buffer = buffer_mgr->CreateBufferTensor("kv_cache_buffer", buffers_shape_map["kv_cache_buffer"], TYPE_FP32,
                                                   MemoryLocation::LOCATION_DEVICE, stream);

  // mtp_hidden_buffer_tensors will used across main forward and nextn forward
  TensorBuffer* mtp_hidden_buffer =
      buffer_mgr->CreateBufferTensor("mtp_hidden_buffer_tensors", buffers_shape_map["mtp_hidden_buffer_tensors"],
                                     weight_type, MemoryLocation::LOCATION_DEVICE, stream);
  mtp_hidden_buffer_tensors = mtp_hidden_buffer->GetTensors();

  StreamSynchronize(*stream);
}

void ModelBuffers::Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config,
                        const RuntimeConfig& runtime_config, BufferManager* buffer_mgr) {
  buffers_ = std::make_unique<ForwardingBuffers>();
  buffers_->Init(context, rank, model_config, runtime_config, buffer_mgr);

  const DataType weight_type = model_config.weight_data_type;

  int rotary_embedding = model_config.rope_scaling_factor_config.use_deepseek_rope
                             ? model_config.mla_config.qk_rope_head_dim
                             : model_config.rotary_embedding;
  int max_position_embeddings = model_config.max_position_embeddings;
  float scale_factor = model_config.rope_scaling_factor_config.factor;

  std::unordered_set<std::string> possible_rope_types = {"su", "longrope", "llama3"};
  if (model_config.use_dsv4_attn) {
    // DSV4: cos_sin_cache uses default rope (for compress_ratio==0 layers),
    //       compress_cos_sin_cache uses yarn rope (for compress_ratio>0 layers).
    TensorBuffer* cos_sin_cache_buffer = buffer_mgr->CreateBufferTensor(
        "cos_sin_cache_tensor_", {static_cast<size_t>(rotary_embedding), static_cast<size_t>(max_position_embeddings)},
        weight_type);
    cos_sin_cache_tensor_ = cos_sin_cache_buffer->GetTensors()[0];

    max_position_embeddings = model_config.rope_scaling_factor_config.original_max_position_embeddings;
    TensorBuffer* compress_cos_sin_cache_buffer = buffer_mgr->CreateBufferTensor(
        "compress_cos_sin_cache_tensor_",
        {static_cast<size_t>(rotary_embedding),
         static_cast<size_t>(max_position_embeddings) * static_cast<size_t>(scale_factor)},
        weight_type);
    compress_cos_sin_cache_tensor_ = compress_cos_sin_cache_buffer->GetTensors()[0];
  } else if (possible_rope_types.find(model_config.rope_scaling_factor_config.type) == possible_rope_types.end() &&
             !model_config.rope_scaling_factor_config.has_alpha) {
    if (model_config.rope_scaling_factor_config.type == "yarn") {
      max_position_embeddings = model_config.rope_scaling_factor_config.original_max_position_embeddings;
    }
    TensorBuffer* cos_sin_cache_buffer = buffer_mgr->CreateBufferTensor(
        "cos_sin_cache_tensor_",
        {static_cast<size_t>(rotary_embedding),
         static_cast<size_t>(max_position_embeddings) * static_cast<size_t>(scale_factor)},
        weight_type);
    cos_sin_cache_tensor_ = cos_sin_cache_buffer->GetTensors()[0];
  } else if (model_config.rope_scaling_factor_config.type == "xdrope") {
    TensorBuffer* cos_sin_cache_buffer = buffer_mgr->CreateBufferTensor(
        "cos_sin_cache_tensor_",
        {static_cast<size_t>(rotary_embedding),
         static_cast<size_t>(max_position_embeddings * model_config.rope_scaling_factor_config.scaling_alpha)},
        weight_type);
    cos_sin_cache_tensor_ = cos_sin_cache_buffer->GetTensors()[0];
  } else {
    TensorBuffer* cos_sin_cache_buffer = buffer_mgr->CreateBufferTensor(
        "cos_sin_cache_tensor_", {static_cast<size_t>(rotary_embedding), static_cast<size_t>(max_position_embeddings)},
        weight_type);
    cos_sin_cache_tensor_ = cos_sin_cache_buffer->GetTensors()[0];
  }
}

void ForwardingContext::Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config,
                             const RuntimeConfig& runtime_config, const PipelineConfig& pipeline_config,
                             ForwardingBuffers* buffers, BufferManager* buffer_mgr,
                             std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  pipeline_config_ = pipeline_config;
  context_ = context;
  rank_ = rank;
  attn_data_parallel_size_ = runtime_config.parallel_basic_config.attn_data_parallel_size;
  buffers_ = buffers;

  vocab_size_ = model_config.vocab_size;
  vocab_size_pad_ = DivRoundUp(model_config.vocab_size, runtime_config.parallel_basic_config.tensor_parallel_size) *
                    runtime_config.parallel_basic_config.tensor_parallel_size;
  const DataType weight_type = model_config.weight_data_type;

  int head_num = model_config.head_num;
  int size_per_head = model_config.size_per_head;
  int hidden_units = size_per_head * head_num;
  int tensor_para_size = runtime_config.parallel_basic_config.tensor_parallel_size;

  size_t max_token_num = runtime_config.max_step_token_num;
  size_t max_batch_size = runtime_config.max_batch_size;
  KLLM_LOG_DEBUG << fmt::format("Max Batch Size = {}, Max Seq Len = {}, Max Token Num = {}",
                                runtime_config.max_batch_size, runtime_config.max_seq_len, max_token_num);

  // TODO(karlluo): we needn't tensor's shape to transfer attribute
  TensorBuffer* forward_shape_buffer = buffer_mgr->CreateBufferTensor("forward_shape", {1}, TYPE_INVALID);
  attn_ctx_.forward_shape = forward_shape_buffer->GetTensors()[0];

  block_cache_layout_ = block_cache_layout;
  // slot 0：始终分配，慢路径与 build-ahead 偶数步共用。
  model_inputs_[0] = std::make_shared<ModelInput>(model_config, runtime_config, rank_, context_, block_cache_layout_);

  // slot 1：仅 FastPathController 启动期 gate 通过时分配；slot 间物理隔离，不复用 buffer。
  if (FastPathController::GetInstance().IsEnabledAtStartup()) {
    model_inputs_[1] = std::make_shared<ModelInput>(model_config, runtime_config, rank_, context_, block_cache_layout_);
    context_->device->EnableDualH2DStreams(true);
    KLLM_LOG_INFO << "ForwardingContext rank=" << rank_ << " allocated dual ModelInput + dual H2D streams (fast path).";
  }
  active_slot_ = 0;

  attn_ctx_.flag_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_BOOL, {1});

  BatchSchedulerConfig batch_scheduler_config;
  Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
  const size_t max_logits_tokens = max_batch_size * batch_scheduler_config.max_decode_tokens_per_req;
  model_output_ = std::make_shared<ModelOutput>(max_logits_tokens, vocab_size_pad_, rank_, context_,
                                                max_token_num * hidden_units, weight_type);

  // Model communicator is only required when tp size is greater than 1.
  if (tensor_para_size > 1) {
    CREATE_BUFFER_SCOPE(reduce_buffer_tensors, buffers_->shared_buffer);
    model_communicator_ = std::make_shared<ModelCommunicator>(
        /* input */ &(reduce_buffer_tensors[0]), rank_, runtime_config, context_);
  } else {
    model_communicator_ = nullptr;
  }
}

void ForwardingContext::SetActiveModelInputSlot(size_t slot) {
  KLLM_CHECK_WITH_INFO(slot < model_inputs_.size(),
                       fmt::format("Invalid ModelInput slot {} (max {}).", slot, model_inputs_.size()));
  KLLM_CHECK_WITH_INFO(model_inputs_[slot] != nullptr,
                       fmt::format("ModelInput slot {} is not allocated (fast path likely disabled).", slot));
  active_slot_ = slot;
}

void ForwardingContext::PrepareModelInputGpu(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode,
                                             size_t model_input_slot) {
  // build-ahead 启用时可传 slot=1；slot 未分配时 hard fail，避免 silent skip。
  KLLM_CHECK_WITH_INFO(
      model_input_slot < model_inputs_.size(),
      fmt::format("PrepareModelInputGpu invalid slot {} (max {}).", model_input_slot, model_inputs_.size()));
  KLLM_CHECK_WITH_INFO(
      model_inputs_[model_input_slot] != nullptr,
      fmt::format("PrepareModelInputGpu slot {} not allocated (enable_executor_build_ahead config mismatch).",
                  model_input_slot));
  model_inputs_[model_input_slot]->SetParseSlot(model_input_slot);
  model_inputs_[model_input_slot]->ParseFromRequests(forward_reqs, run_mode);
}

void ForwardingContext::UpdateBeforeForward(std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode,
                                            size_t model_input_slot, bool gpu_input_already_prepared) {
  PROFILE_EVENT_SCOPE(UpdateBeforeForward, "UpdateBeforeForward", rank_);

  // 慢路径 model_input_slot 恒为 0；build-ahead 传入实际 slot。
  // slot 1 须在构造函数按相同条件分配，缺失即配置不一致，hard fail。
  KLLM_CHECK_WITH_INFO(
      model_input_slot < model_inputs_.size(),
      fmt::format("UpdateBeforeForward invalid slot {} (max {}).", model_input_slot, model_inputs_.size()));
  KLLM_CHECK_WITH_INFO(
      model_inputs_[model_input_slot] != nullptr,
      fmt::format("UpdateBeforeForward slot {} not allocated (enable_executor_build_ahead config mismatch).",
                  model_input_slot));
  active_slot_ = model_input_slot;
  auto& model_input = model_inputs_[active_slot_];

  if (!gpu_input_already_prepared) {
    model_input->SetParseSlot(model_input_slot);
    model_input->ParseFromRequests(forward_reqs, run_mode);
  }

  // create forward shape tensor
  attn_ctx_.forward_shape.shape = {
      model_input->multi_token_request_num,          // request num use flash_attention kernel
      model_input->multi_token_request_max_tokens,   // max request tokens of request that use flash_attention kernel
      model_input->context_kv_cache_block_num,       // total kv cache block num that use flash_attention kernel
      model_input->single_token_request_num,         // request num use page_attention kernel
      model_input->single_token_request_max_tokens,  // max request tokens of request that use page_attention kernel
      model_input->decode_kv_cache_block_num,        // total kv cache block num that use page_attention kernel
      model_input->dp_max_forwarding_tokens,         // used for blocked_prefill
      model_input->dp_multi_token_request_num,
      model_input->dp_multi_token_request_max_tokens,
      model_input->dp_single_token_request_num,
      model_input->dp_single_token_request_max_tokens,
      model_input->dp_total_prefix_len,
      model_input->dp_decode_tokens};  // total decode token count (may differ from single_token_request_num in MTP)
#ifdef ENABLE_ACL
  attn_ctx_.forward_shape.shape = {
      std::max(model_input->multi_token_request_num, model_input->single_token_request_num),
      std::max(model_input->multi_token_request_max_tokens, model_input->single_token_request_max_tokens),
      model_input->context_kv_cache_block_num + model_input->decode_kv_cache_block_num};
#endif
  // Pass the `use_cache` flag to `flag_tensor_`.
  attn_ctx_.flag_tensor.template GetPtr<bool>()[0] = model_input->use_cache;
}

void ForwardingContext::UpdateAfterForward(std::vector<ForwardRequest>& forward_reqs) {
  // Cast to float & Copy to logits buffer
  attn_ctx_.forward_shape.shape = {forward_reqs[0].logits_offset * vocab_size_ * sizeof(float), vocab_size_,
                                   vocab_size_pad_};
}

}  // namespace ksana_llm
