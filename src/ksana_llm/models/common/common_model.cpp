/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/common/common_model.h"

#include <memory>
#include <vector>

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/singleton.h"
#include "torch/csrc/autograd/python_variable.h"

namespace ksana_llm {

template <typename T>
CommonModel<T>::CommonModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context) {
  model_config_ = model_config;
  context_ = context;
  rank_ = rank;
}

template <typename T>
CommonModel<T>::~CommonModel() {}

template <typename T>
void CommonModel<T>::InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight) {
  GetBlockManager()->SetDeviceId(rank_);

  prefix_caching_enabled_ = Singleton<Environment>::GetInstance()->IsPrefixCachingEnabled();

  model_run_config_ = model_run_config;

  num_layer_ = model_config_.num_layer;
  vocab_size_ = model_config_.vocab_size;
  vocab_size_pad_ =
      DivRoundUp(model_config_.vocab_size, model_config_.tensor_para_size) * model_config_.tensor_para_size;

  int head_num = model_config_.head_num;
  int size_per_head = model_config_.size_per_head;
  int hidden_units = size_per_head * head_num;
  int tensor_para_size = model_config_.tensor_para_size;
  int rotary_embedding = model_config_.rotary_embedding;
  int head_num_per_tp = head_num / tensor_para_size;
  int num_kv_heads_per_tp = model_config_.num_key_value_heads / tensor_para_size;
  int stride_size = (head_num_per_tp + num_kv_heads_per_tp * 2) * size_per_head;
  int max_position_embeddings = model_config_.max_position_embeddings;
  float rope_theta = model_config_.rope_theta;

  BlockManagerConfig block_manager_config;
  STATUS_CHECK_FAILURE(Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config));

  size_t max_token_num = model_config_.max_scheduler_token_num;
  size_t max_batch_size = model_config_.max_batch_size;
  KLLM_LOG_DEBUG << fmt::format("Max Batch Size = {}, Max Seq Len = {}, Max Token Num = {}",
                                model_config_.max_batch_size, model_config_.max_token_num, max_token_num);

  int inter_size_per_tp = model_config_.inter_size / tensor_para_size;
  int max_dim =
      std::max(std::max((head_num_per_tp + 2 * num_kv_heads_per_tp) * size_per_head, hidden_units), inter_size_per_tp);
  size_t hidden_buffer_size = std::max(max_batch_size * vocab_size_pad_, max_token_num * max_dim);
  size_t residual_buffer_size = max_token_num * hidden_units;
  // `shared_buffer_` is shared by `gated_buffer_`, `reduce_buffer_` and `paged_buffer_`.
  size_t shared_buffer_size = max_token_num * std::max(inter_size_per_tp, hidden_units * 2);

  DataType weight_type = model_config_.weight_data_type;
  // TODO(karlluo): all create tensor used dynamic memory pool
  STATUS_CHECK_FAILURE(CreateBufferTensor(hidden_buffer_0_[0], {hidden_buffer_size}, weight_type));
  STATUS_CHECK_FAILURE(CreateBufferTensor(hidden_buffer_1_[0], {hidden_buffer_size}, weight_type));
  STATUS_CHECK_FAILURE(CreateBufferTensor(residual_buffer_[0], {residual_buffer_size}, weight_type));
  STATUS_CHECK_FAILURE(CreateBufferTensor(shared_buffer_[0], {shared_buffer_size}, weight_type));

  STATUS_CHECK_FAILURE(CreateBufferTensor(
      cos_sin_cache_tensor_, {static_cast<size_t>(rotary_embedding), static_cast<size_t>(max_position_embeddings)},
      weight_type));
#ifdef ENABLE_ACL
  STATUS_CHECK_FAILURE(
      CreateBufferTensor(ascend_buffer_0_, {max_token_num, static_cast<size_t>(hidden_units)}, TYPE_FP16));
  STATUS_CHECK_FAILURE(
      CreateBufferTensor(ascend_buffer_1_, {max_token_num, static_cast<size_t>(hidden_units)}, TYPE_FP16));
  STATUS_CHECK_FAILURE(
      CreateBufferTensor(ascend_buffer_2_, {max_token_num, static_cast<size_t>(hidden_units)}, TYPE_FP16));
  STATUS_CHECK_FAILURE(
      CreateBufferTensor(ascend_buffer_3_, {max_token_num, static_cast<size_t>(hidden_units)}, TYPE_FP16));
  STATUS_CHECK_FAILURE(
      CreateBufferTensor(ascend_buffer_4_, {max_token_num, static_cast<size_t>(hidden_units)}, TYPE_FP16));

  for (int idx = 0; idx < num_layer_; ++idx) {
    Tensor key_cache;
    Tensor val_cache;
    STATUS_CHECK_FAILURE(CreateBufferTensor(key_cache, {max_token_num, static_cast<size_t>(hidden_units)}, TYPE_FP16));
    STATUS_CHECK_FAILURE(CreateBufferTensor(val_cache, {max_token_num, static_cast<size_t>(hidden_units)}, TYPE_FP16));

    ascend_key_caches_.push_back(key_cache);
    ascend_val_caches_.push_back(val_cache);
  }

#endif
  // TODO(karlluo): we needn't tensor's shape to transfer attribute
  STATUS_CHECK_FAILURE(CreateBufferTensor(forward_shape_, {1}, TYPE_INT32));

  KLLM_LOG_DEBUG << "Total buffer tensors memory used: " << (GetBufferTensorsMemoryUsed() >> 20) << " MB";

  // Initialize instances for each layer.
  emb_lookup_layer_ = std::make_shared<EmbLookupLayer<T>>();
  if (model_run_config_.position_encoding == PositionEncoding::LEARNED_ABSOLUTE) {
    Tensor position_weight = base_weight->GetModelWeights("model.embed_positions.weight");
    emb_lookup_layer_->Init({static_cast<T>(model_run_config_.emb_scale), position_weight.GetPtr<void>()}, context_,
                            rank_);
  } else {
    emb_lookup_layer_->Init({}, context_, rank_);
  }

  cpu_emb_lookup_layer_ = std::make_shared<CpuEmbLookupLayer<T>>();
  cpu_emb_lookup_layer_->Init({}, context_, rank_);

  layernorm_layer_ = std::make_shared<LayernormLayer<T>>();
  layernorm_layer_->Init({model_config_.layernorm_eps}, context_, rank_);

  add_layer_ = std::make_shared<AddLayer<T>>();
  add_layer_->Init({}, context_, rank_);

  silu_mul_layer_ = std::make_shared<SiluMulLayer<T>>();
  silu_mul_layer_->Init({}, context_, rank_);

  // create matmul layer
  CreateProjLayer(base_weight);

  assemble_last_token_layer_ = std::make_shared<AssembleLastTokenLayer<T>>();
  assemble_last_token_layer_->Init({}, context_, rank_);

  cast_layer_ = std::make_shared<CastLayer<T>>();
  cast_layer_->Init({}, context_, rank_);

  input_refit_layer_ = std::make_shared<InputRefitLayer<T>>();
  input_refit_layer_->Init({}, context_, rank_);

  model_input_ = std::make_shared<ModelInput>(model_config_, rank_, context_);

  if (Singleton<Environment>::GetInstance()->EmbedTokensUseCpu()) {
    CreateTensor(cpu_input_tokens_tensor_, model_input_->input_ids.shape, model_input_->input_ids.dtype, rank_,
                 MemoryDevice::MEMORY_HOST);
    CreateTensor(cpu_tokens_emb_tensor_, {model_input_->input_ids.shape[0] * hidden_units},
                 model_input_->input_ids.dtype, rank_, MemoryDevice::MEMORY_HOST);
  }
  model_output_ = std::make_shared<ModelOutput>(max_batch_size, vocab_size_pad_, rank_, context_);

  // Model communicator is only required when tp size is greater than 1.
  if (tensor_para_size > 1) {
    // Currently, custom all reduce is only enabled when `tp == 2`, so the `hidden_buffer_0_` will not be used.
    model_communicator_ = std::make_shared<ModelCommunicator<T>>(/* buffer */ &hidden_buffer_0_[0],
                                                                 /* input */ &reduce_buffer_[0], rank_, context_);
  } else {
    model_communicator_ = nullptr;
  }

  flash_attention_layers_.resize(num_layer_);
  paged_attention_layers_.resize(num_layer_);
  void* cos_sin_cache_ptr = cos_sin_cache_tensor_.GetPtr<void>();
  for (int idx = 0; idx < num_layer_; ++idx) {
    flash_attention_layers_[idx] = CreateAttentionLayer<T, FlashAttentionLayer>(GetBlockManager()->GetDtype());
    paged_attention_layers_[idx] = CreateAttentionLayer<T, PagedAttentionLayer>(GetBlockManager()->GetDtype());
    // NOTE(karlluo): acsends's image g++ is 9.4.0, it do not support convert from ‘<brace-enclosed initializer list>’
    // to ‘std::vector<std::any>’ so we use push back to make it work.
    std::vector<std::any> attention_param;
    attention_param.push_back(idx);
    attention_param.push_back(num_layer_);
    attention_param.push_back(max_position_embeddings);
    attention_param.push_back(head_num_per_tp);
    attention_param.push_back(num_kv_heads_per_tp);
    attention_param.push_back(size_per_head);
    attention_param.push_back(stride_size);
    attention_param.push_back(tensor_para_size);
    attention_param.push_back(GetBlockManager()->GetDtype());
    attention_param.push_back(model_config_.k_scales[idx]);
    attention_param.push_back(model_config_.v_scales[idx]);
    attention_param.push_back(rotary_embedding);
    attention_param.push_back(rope_theta);
    attention_param.push_back(model_run_config_.is_neox);
    attention_param.push_back(model_run_config_.position_encoding);
    attention_param.push_back(std::any(cos_sin_cache_ptr));
    attention_param.push_back(model_config_.rope_scaling_factor_config);
    attention_param.push_back(max_batch_size);
    std::vector<std::any> flash_attention_param = attention_param;
    std::vector<std::any> paged_attention_param = attention_param;
    // NOTE(karlluo): bool for is_context_decode_stage
    flash_attention_param.push_back(true);
    paged_attention_param.push_back(false);
    flash_attention_layers_[idx]->Init(flash_attention_param, context_, rank_);
    paged_attention_layers_[idx]->Init(paged_attention_param, context_, rank_);
  }

  // search optional plugin
  auto optional_file = Singleton<OptionalFile>::GetInstance();
  std::string& plugin_path =
      optional_file->GetOptionalFile(model_config_.path, "ksana_plugin/" + plugin_name, "ksana_plugin.py");
  // try to load plugin
  py::gil_scoped_acquire acquire;
  try {
    py::module importlib_util = py::module::import("importlib.util");
    py::object spec = importlib_util.attr("spec_from_file_location")("ksana_plugin", plugin_path);
    py::object module = importlib_util.attr("module_from_spec")(spec);
    spec.attr("loader").attr("exec_module")(module);

    plugin_ = std::make_shared<py::object>(module.attr("KsanaPlugin")());

    KLLM_LOG_INFO << "Using Plugin";
  } catch (const py::error_already_set& e) {
    PyErr_Clear();
  }
  // if load plugin success, try to init plugin
  if (plugin_) {
    py::dict kwargs;
    kwargs["model_path"] = model_config_.path;
    kwargs["preprocess"] = true;
    plugin_->attr("init_plugin")(**kwargs);
  }
}

template <typename T>
float* CommonModel<T>::GetLogitsPtr() {
  GetBlockManager()->SetDeviceId(rank_);
  return model_output_->logits_tensor.GetPtr<float>();
}

template <typename T>
Status CommonModel<T>::CreateProjLayer(std::shared_ptr<BaseWeight>& base_weight) {
  DataType weight_type = model_config_.weight_data_type;
  DataType input_type = weight_type;
  DataType output_type = weight_type;

  // auto create matmul layers
  matmul_layer_factory_ =
      std::make_shared<MatMulLayerFactory<T>>(shared_matmul_workspace_buffer_, model_config_, rank_, context_);

  attn_qkv_proj_layer_ = matmul_layer_factory_->AutoCreateLayer(
      base_weight, "model.layers.0.self_attn.query_key_value.weight", weight_type, input_type, output_type, {});
  attn_o_proj_layer_ = matmul_layer_factory_->AutoCreateLayer(base_weight, "model.layers.0.self_attn.o_proj.weight",
                                                              weight_type, input_type, output_type, {});
  mlp_gate_proj_layer_ = matmul_layer_factory_->AutoCreateLayer(base_weight, "model.layers.0.mlp.gate_proj.weight",
                                                                weight_type, input_type, output_type, {});
  // Only gated activation has up_proj.
  if (base_weight->GetModelWeights("model.layers.0.mlp.up_proj.weight").GetBlockId() >= 0) {
    mlp_up_proj_layer_ = matmul_layer_factory_->AutoCreateLayer(base_weight, "model.layers.0.mlp.up_proj.weight",
                                                                weight_type, input_type, output_type, {});
  } else {
    mlp_up_proj_layer_ = nullptr;
  }
  mlp_down_proj_layer_ = matmul_layer_factory_->AutoCreateLayer(base_weight, "model.layers.0.mlp.down_proj.weight",
                                                                weight_type, input_type, output_type, {});
  lm_head_proj_layer_ =
      matmul_layer_factory_->AutoCreateLayer(base_weight, "lm_head.weight", weight_type, input_type, output_type, {});

#ifdef ENABLE_ACL
  attn_qkv_proj_layer_->Init({}, context_, rank_);
  attn_o_proj_layer_->Init({}, context_, rank_);
  mlp_gate_proj_layer_->Init({}, context_, rank_);
  if (mlp_up_proj_layer_) {
    mlp_up_proj_layer_->Init({}, context_, rank_);
  }
  mlp_down_proj_layer_->Init({}, context_, rank_);
  lm_head_proj_layer_->Init({}, context_, rank_);
#endif

  return Status();
}

template <typename T>
Status CommonModel<T>::AddAttentionPrefixCache() {
  // Before MMHA inference, retrieve the key-value pairs (k, v) from the Prefix Cache Block
  // and populate them into the mmha prefix input tensor.
  auto& mmha_origin_input = hidden_buffer_0_[0];
  auto& mmha_prefix_input = hidden_buffer_1_[0];

  size_t total_token_num = 0;
  size_t dtype_size = GetTypeSize(mmha_origin_input.dtype);
  size_t size_per_token = mmha_origin_input.shape[1] * dtype_size;
  for (size_t idx = 0; idx < model_input_->batch_size; ++idx) {
    size_t src_offset = (model_input_->input_offset_list[idx] - model_input_->input_prefix_list[idx]) * size_per_token;
    size_t input_length = model_input_->input_offset_list[idx + 1] - model_input_->input_offset_list[idx];
    size_t prefix_length = model_input_->input_prefix_list[idx + 1] - model_input_->input_prefix_list[idx];
    size_t copy_size = (input_length - prefix_length) * size_per_token;
    size_t dst_offset = (model_input_->input_offset_list[idx] + prefix_length) * size_per_token;
    if (idx >= model_input_->context_num && model_input_->decode_num > 0) {
      MemcpyAsync(shared_buffer_[0].GetPtr<void>(), mmha_origin_input.GetPtr<void>() + src_offset,
                  copy_size * model_input_->decode_num, MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
      shared_buffer_[0].shape = {copy_size / dtype_size, model_input_->decode_num};
      total_token_num += model_input_->decode_num;
      break;
    }
    MemcpyAsync(mmha_prefix_input.GetPtr<void>() + dst_offset, mmha_origin_input.GetPtr<void>() + src_offset, copy_size,
                MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
    total_token_num += input_length;
  }
  mmha_prefix_input.shape = {total_token_num, mmha_origin_input.shape[1]};
  mmha_prefix_input.dtype = mmha_origin_input.dtype;
  StreamSynchronize(context_->GetComputeStreams()[rank_]);

  std::swap(hidden_buffer_1_, hidden_buffer_0_);
  return Status();
}

template <typename T>
Status CommonModel<T>::RemoveAttentionPrefixCache() {
  // After the completion of MMHA inference, copy the data from the MMHA output results,
  // excluding the Prefix Cache section, and continue with the subsequent inference.
  auto& mmha_prefix_output = hidden_buffer_0_[0];
  auto& mmha_output = hidden_buffer_1_[0];
  auto attention_input_shape = hidden_buffer_1_[0].shape;
  size_t total_token_num_without_prefix = 0;
  size_t dst_offset = 0;
  size_t src_offset = 0;
  size_t dtype_size = GetTypeSize(mmha_prefix_output.dtype);
  size_t size_per_token = mmha_prefix_output.shape[1] * dtype_size;
  for (size_t idx = 0; idx < model_input_->context_num; ++idx) {
    size_t prefix_length = model_input_->input_prefix_list[idx + 1] - model_input_->input_prefix_list[idx];
    size_t input_length = model_input_->input_offset_list[idx + 1] - model_input_->input_offset_list[idx];
    src_offset += prefix_length * size_per_token;
    size_t copy_size = size_per_token * (input_length - prefix_length);

    MemcpyAsync(mmha_output.GetPtr<void>() + dst_offset, mmha_prefix_output.GetPtr<void>() + src_offset, copy_size,
                MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
    src_offset += copy_size;
    dst_offset += copy_size;
    total_token_num_without_prefix += input_length - prefix_length;
  }
  mmha_output.shape = {total_token_num_without_prefix, mmha_prefix_output.shape[1]};
  mmha_output.dtype = mmha_prefix_output.dtype;
  if (model_input_->decode_num > 0) {
    MemcpyAsync(hidden_buffer_0_[0].GetPtr<void>() +
                    shared_buffer_[0].GetTotalBytes() / model_input_->decode_num * total_token_num_without_prefix,
                shared_buffer_[0].GetPtr<void>(), shared_buffer_[0].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
                context_->GetComputeStreams()[rank_]);
    hidden_buffer_0_[0].shape = {total_token_num_without_prefix + model_input_->decode_num, attention_input_shape[1]};
  }
  StreamSynchronize(context_->GetComputeStreams()[rank_]);

  std::swap(hidden_buffer_1_, hidden_buffer_0_);
  return Status();
}

template <typename T>
Status CommonModel<T>::FlashAttentionForward(const int layer_idx) {
  bool reuse_prefix_caching = prefix_caching_enabled_;

  if (reuse_prefix_caching) {
    AddAttentionPrefixCache();
  }

#ifdef ENABLE_CUDA
  STATUS_CHECK_RETURN(flash_attention_layers_[layer_idx]->Forward(
      {hidden_buffer_0_[0], model_input_->input_offset_uint64_tensor, model_input_->kv_list,
       model_input_->input_prefix_uint64_tensor, model_input_->kv_cache_offset_tensor,
       model_input_->rotary_embedding_pos, model_input_->rotary_embedding_mask, forward_shape_},
      hidden_buffer_1_));
#elif defined(ENABLE_ACL)
  // inference on NPU with ATB
  STATUS_CHECK_RETURN(flash_attention_layers_[layer_idx]->Forward(
      {hidden_buffer_0_[0], model_input_->rotary_embedding_pos, model_input_->layers_slot_mapping,
       model_input_->k_cache_blocks_base, model_input_->v_cache_blocks_base, model_input_->seq_len_host, forward_shape_,
       model_input_->atb_attention_attr},
      hidden_buffer_1_));
#endif

  std::swap(hidden_buffer_1_, hidden_buffer_0_);

  if (reuse_prefix_caching) {
    RemoveAttentionPrefixCache();
  }

  return Status();
}

template <typename T>
Status CommonModel<T>::PagedAttentionForward(const int layer_idx) {
#ifdef ENABLE_CUDA
  STATUS_CHECK_RETURN(paged_attention_layers_[layer_idx]->Forward(
      {hidden_buffer_0_[0], model_input_->input_tokens_int32_tensor, model_input_->kv_list,
       model_input_->kv_cache_offset_tensor, model_input_->rotary_embedding_pos, model_input_->rotary_embedding_mask,
       model_input_->kv_cache_buffer, forward_shape_, /* workspace */ paged_buffer_[0]},
      hidden_buffer_1_));
#elif defined(ENABLE_ACL)
  // inference on NPU with ATB
  STATUS_CHECK_RETURN(paged_attention_layers_[layer_idx]->Forward(
      {hidden_buffer_0_[0], model_input_->rotary_embedding_pos, model_input_->layers_slot_mapping,
       model_input_->layers_block_table, model_input_->k_cache_blocks_base, model_input_->v_cache_blocks_base,
       model_input_->seq_len_host, forward_shape_, model_input_->atb_attention_attr},
      hidden_buffer_1_));
#endif
  std::swap(hidden_buffer_1_, hidden_buffer_0_);
  return Status();
}

template <typename T>
Status CommonModel<T>::LayerNormForward(const std::string& layer_name,
                                        std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                        const std::vector<Tensor>& layernorm_input,
                                        std::vector<Tensor>& layernorm_output) {
  Tensor layernorm_weight = base_weight->GetModelWeights(layer_name);
  STATUS_CHECK_RETURN(layernorm_layer_->Forward({layernorm_input[0], layernorm_weight}, layernorm_output));
  return Status();
}

template <typename T>
Status CommonModel<T>::CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                       const std::vector<Tensor>& attention_input, const bool is_context_stage) {
  // Attn proj MatMul
  Tensor attn_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.query_key_value.weight", layer_idx));
  STATUS_CHECK_RETURN(attn_qkv_proj_layer_->Forward({attention_input[0], attn_proj_weight}, hidden_buffer_1_));
  if (model_run_config_.qkv_add_bias) {
    Tensor attn_proj_bias =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.query_key_value.bias", layer_idx));
    STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_1_[0], attn_proj_bias}, hidden_buffer_1_));
  }
  std::swap(hidden_buffer_1_, hidden_buffer_0_);

  // MMHA Flash/Paged Attention
  if (layer_idx == 0) {
    // only need sync in the first layer
    StreamWaitEvent(context_->GetComputeStreams()[rank_], model_input_->kvcache_offset_event);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], model_input_->rotary_embedding_event);
  }
  if (model_input_->context_num) {
    FlashAttentionForward(layer_idx);
    if (model_input_->decode_num) {
      std::swap(hidden_buffer_1_, hidden_buffer_0_);
    }
  }
  if (model_input_->decode_num) {
    PagedAttentionForward(layer_idx);
  }

  // Attn o_proj MatMul
  Tensor attn_o_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx));
  if (model_communicator_) {
    // Put output to `reduce_buffer_` to ensure that the input for custom reduce sum is always in `reduce_buffer_`.
    STATUS_CHECK_RETURN(attn_o_proj_layer_->Forward({hidden_buffer_0_[0], attn_o_proj_weight}, reduce_buffer_));
  } else {
    STATUS_CHECK_RETURN(attn_o_proj_layer_->Forward({hidden_buffer_0_[0], attn_o_proj_weight}, hidden_buffer_1_));
    std::swap(hidden_buffer_1_, hidden_buffer_0_);
  }

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }

  // Attn AllReduceSum
  if (model_communicator_) {
    model_communicator_->ReduceSum(reduce_buffer_, hidden_buffer_0_, is_context_stage, true);
  }
  return Status();
}

template <typename T>
Status CommonModel<T>::CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                 const std::vector<Tensor>& mlp_input, const bool is_context_stage) {
  // Mlp gate_proj MatMul
  Tensor gate_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.gate_proj.weight", layer_idx));
  STATUS_CHECK_RETURN(mlp_gate_proj_layer_->Forward({mlp_input[0], gate_proj_weight}, hidden_buffer_1_));
  // Mlp up_proj MatMul 由于 gate_proj 与 up_proj 为并行关系,因此此处使用额外空间存储 matmul 结果
  Tensor up_proj_weight = base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.up_proj.weight", layer_idx));
  STATUS_CHECK_RETURN(mlp_up_proj_layer_->Forward({mlp_input[0], up_proj_weight}, gated_buffer_));
  std::swap(hidden_buffer_1_, hidden_buffer_0_);

  // Activation is an in-place operation, just put the output in `hidden_buffer_0_`, the same as the input.
  STATUS_CHECK_RETURN(silu_mul_layer_->Forward({hidden_buffer_0_[0], gated_buffer_[0]}, hidden_buffer_0_));

  // Mlp down_proj MatMul
  Tensor down_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.down_proj.weight", layer_idx));
  if (model_communicator_) {
    // Put output to `reduce_buffer_` to ensure that the input for custom reduce sum is always in `reduce_buffer_`.
    STATUS_CHECK_RETURN(mlp_down_proj_layer_->Forward({hidden_buffer_0_[0], down_proj_weight}, reduce_buffer_));
  } else {
    STATUS_CHECK_RETURN(mlp_down_proj_layer_->Forward({hidden_buffer_0_[0], down_proj_weight}, hidden_buffer_1_));
    std::swap(hidden_buffer_1_, hidden_buffer_0_);
  }

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }

  // Mlp AllReduceSum
  if (model_communicator_) {
    model_communicator_->ReduceSum(reduce_buffer_, hidden_buffer_0_, is_context_stage, true);
  }
  return Status();
}

template <typename T>
Status CommonModel<T>::CommonDecoderPreNorm(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                            const bool is_context_stage) {
  KLLM_CHECK_WITH_INFO(model_run_config_.layernorm_position == LayerNormPosition::PRE_NORM,
                       "CommonDecoderPreNorm should be called by pre norm models");

  // Pre attn layernorm
  // Pre layernorm uses layernorm input for residual connection.
  LayerNormForward(fmt::format("model.layers.{}.input_layernorm.weight", layer_idx), base_weight, residual_buffer_,
                   hidden_buffer_0_);

  // Common attention
  STATUS_CHECK_RETURN(CommonAttention(layer_idx, base_weight, hidden_buffer_0_, is_context_stage));

  // Attn residual add
  STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_0_[0], residual_buffer_[0]}, residual_buffer_));

  // Pre mlp layernorm
  // Pre layernorm uses layernorm input for residual connection.
  LayerNormForward(fmt::format("model.layers.{}.post_attention_layernorm.weight", layer_idx), base_weight,
                   residual_buffer_, hidden_buffer_0_);

  // Common mlp
  STATUS_CHECK_RETURN(CommonMlp(layer_idx, base_weight, hidden_buffer_0_, is_context_stage));

  // Mlp residual add
  STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_0_[0], residual_buffer_[0]}, residual_buffer_));

  return Status();
}

template <typename T>
Status CommonModel<T>::CommonDecoderPostNorm(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                             const bool is_context_stage) {
  KLLM_CHECK_WITH_INFO(model_run_config_.layernorm_position == LayerNormPosition::POST_NORM,
                       "CommonDecoderPostNorm should be called by post norm models");

  // Common attention
  // Post layernorm uses attention input for residual connection.
  STATUS_CHECK_RETURN(CommonAttention(layer_idx, base_weight, residual_buffer_, is_context_stage));

  // Attn residual add
  STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_0_[0], residual_buffer_[0]}, hidden_buffer_0_));

  // Post attn layernorm
  LayerNormForward(fmt::format("model.layers.{}.input_layernorm.weight", layer_idx), base_weight, hidden_buffer_0_,
                   residual_buffer_);

  // Common mlp
  // Post layernorm uses mlp input for residual connection.
  STATUS_CHECK_RETURN(CommonMlp(layer_idx, base_weight, residual_buffer_, is_context_stage));

  // Mlp residual add
  STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_0_[0], residual_buffer_[0]}, hidden_buffer_0_));

  // Post mlp layernorm
  LayerNormForward(fmt::format("model.layers.{}.post_attention_layernorm.weight", layer_idx), base_weight,
                   hidden_buffer_0_, residual_buffer_);
  return Status();
}

template <typename T>
Status CommonModel<T>::EmbedTokensUseCpu(Tensor& embedding_weight, std::vector<ForwardRequest>& forward_reqs) {
  auto batch_size = forward_reqs.size();
  void* input_tokens_ptr = cpu_input_tokens_tensor_.GetPtr<void>();
  size_t index = 0;
  for (size_t idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    if (forward_reqs[idx].step == 0) {
      size_t copy_len = req_input->size() * sizeof(int);
      memcpy(input_tokens_ptr + index, req_input->data(), copy_len);
      index += copy_len;
    } else {
      memcpy(input_tokens_ptr + index, &req_input->back(), sizeof(int));
      index += sizeof(int);
    }
  }
  cpu_input_tokens_tensor_.shape = {index / sizeof(int)};
  cpu_emb_lookup_layer_->Forward({cpu_input_tokens_tensor_, cpu_tokens_emb_tensor_, embedding_weight},
                                 residual_buffer_);
  return Status();
}

template <typename T>
Status CommonModel<T>::EmbedTokensUseGpu(Tensor& embedding_weight) {
  // Wait the computation of input_ids.
  StreamWaitEvent(context_->GetComputeStreams()[rank_], model_input_->input_ids_event);

  STATUS_CHECK_RETURN(emb_lookup_layer_->Forward({model_input_->input_ids, model_input_->input_offset_uint64_tensor,
                                                  model_input_->input_prefix_uint64_tensor, embedding_weight},
                                                 residual_buffer_));

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }

  if (model_communicator_) {
    model_communicator_->AllGather({residual_buffer_[0], hidden_buffer_1_[0]}, residual_buffer_);
  }
  return Status();
}

template <typename T>
bool CommonModel<T>::UpdateResponse(std::vector<ForwardRequest>& forward_reqs, Tensor& output,
                                    const std::string& stage) {
  bool ret = true;
  int req_offset = 0;
  for (ForwardRequest& req : forward_reqs) {
    int output_tokens_num = req.output_tokens->size();
    req_offset += output_tokens_num;
    if (!req.request_target) {
      ret = false;
      continue;
    }
    auto it = req.request_target->find(stage);
    if (it == req.request_target->end()) {
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
      for (int i = 0; i < output_tokens_num; i++) {
        if (token_id_set.count(req.output_tokens->at(i)) > 0) {
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
    output_len = 0;
    // Copy data from the output tensor to the output_data buffer based on slice positions.
    for (auto [l, r] : slice_pos) {
      MemcpyAsync(ret_tensor.data.data() + output_len * chunk_size,
                  output.GetPtr<void>() + (req_offset - output_tokens_num + l) * chunk_size, (r - l + 1) * chunk_size,
                  MEMCPY_DEVICE_TO_HOST, context_->GetComputeStreams()[rank_]);
      output_len += r - l + 1;
    }
    StreamSynchronize(context_->GetComputeStreams()[rank_]);
  }
  return ret;
}

template <typename T>
Status CommonModel<T>::CommonForward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                     std::vector<ForwardRequest>& forward_reqs) {
  GetBlockManager()->SetDeviceId(rank_);

  // CPU embedding lookup
  // The output is stored in `residual_buffer_` for residual connection in common decoder.
  Tensor embedding_weight = base_weight->GetModelWeights("model.embed_tokens.weight");
  if (embedding_weight.device == MemoryDevice::MEMORY_HOST) {
    EmbedTokensUseCpu(embedding_weight, forward_reqs);
  }

  model_input_->ParseFromRequests(forward_reqs);

  // create forward shape tensor
  forward_shape_.shape = {
      model_input_->context_num, model_input_->context_max_tokens, model_input_->context_total_block_num,
      model_input_->decode_num,  model_input_->decode_max_tokens,  model_input_->decode_total_block_num};
#ifdef ENABLE_ACL
  forward_shape_.shape = {std::max(model_input_->context_num, model_input_->decode_num),
                          std::max(model_input_->context_max_tokens, model_input_->decode_max_tokens),
                          static_cast<size_t>(model_input_->kv_cache_offset_list.back())};
#endif
  bool is_context_stage = model_input_->context_num > 0;

  // GPU embedding lookup
  // The output is stored in `residual_buffer_` for residual connection in common decoder.
  if (embedding_weight.device == MemoryDevice::MEMORY_DEVICE) {
    EmbedTokensUseGpu(embedding_weight);
  }

  // refit input needs to be processed only in the context stage.
  if (is_context_stage) {
    input_refit_layer_->Forward({model_input_->cpu_input_refit_tensor.pos_pair_tensor,
                                 model_input_->cpu_input_refit_tensor.emb_fp32_ptr_tensor},
                                residual_buffer_);
  }

  // CommonDecoder
  if (model_run_config_.layernorm_position == LayerNormPosition::PRE_NORM) {
    for (int layer_idx = 0; layer_idx < num_layer_; ++layer_idx) {
      STATUS_CHECK_RETURN(CommonDecoderPreNorm(layer_idx, base_weight, is_context_stage));
    }
  } else {  // LayerNormPosition::POST_NORM
    for (int layer_idx = 0; layer_idx < num_layer_; ++layer_idx) {
      STATUS_CHECK_RETURN(CommonDecoderPostNorm(layer_idx, base_weight, is_context_stage));
    }
  }

  if (is_context_stage) {
    if (UpdateResponse(forward_reqs, residual_buffer_[0], "transformer")) {
      StreamSynchronize(context_->GetComputeStreams()[rank_]);
      input_refit_layer_->Clear();
      return Status();
    }
  }

  // final norm
  // Only pre norm model performs final norm.
  // Both input and output are in `residual_buffer_`.
  if (model_run_config_.layernorm_position == LayerNormPosition::PRE_NORM) {
    LayerNormForward("model.norm.weight", base_weight, residual_buffer_, residual_buffer_);
  }

  if (is_context_stage) {
    if (UpdateResponse(forward_reqs, residual_buffer_[0], "layernorm")) {
      StreamSynchronize(context_->GetComputeStreams()[rank_]);
      input_refit_layer_->Clear();
      return Status();
    }
  }

  // assemble last token
  // The input is stored in `residual_buffer_`.
  if (model_input_->use_logits_custom_length) {
    STATUS_CHECK_RETURN(
        assemble_last_token_layer_->Forward({residual_buffer_[0], model_input_->logits_custom_length_uint64_tensor,
                                             model_input_->logits_length_prefix_uint64_tensor},
                                            hidden_buffer_0_));
  } else {
#ifdef ENABLE_CUDA
    STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward(
        {residual_buffer_[0], model_input_->input_offset_uint64_tensor, model_input_->input_prefix_uint64_tensor},
        hidden_buffer_0_));
#else
    STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward(
        {residual_buffer_[0], model_input_->last_token_index_tensor, model_input_->input_prefix_uint64_tensor},
        hidden_buffer_0_));
#endif
  }

  // lm_head
  Tensor lm_head_weight = base_weight->GetModelWeights("lm_head.weight");
  STATUS_CHECK_RETURN(lm_head_proj_layer_->Forward({hidden_buffer_0_[0], lm_head_weight}, hidden_buffer_1_));
  std::swap(hidden_buffer_1_, hidden_buffer_0_);

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }

  if (model_communicator_) {
    model_communicator_->AllGather({hidden_buffer_0_[0], hidden_buffer_1_[0]}, hidden_buffer_0_);
  }

  // Cast to float & Copy to logits buffer
  forward_shape_.shape = {forward_reqs[0].logits_offset * vocab_size_ * sizeof(float), vocab_size_, vocab_size_pad_};
  std::vector<Tensor> logits_buffer{model_output_->logits_tensor};
  STATUS_CHECK_RETURN(cast_layer_->Forward({hidden_buffer_0_[0], forward_shape_}, logits_buffer));
  StreamSynchronize(context_->GetComputeStreams()[rank_]);

  input_refit_layer_->Clear();
  return Status();
}

template <typename T>
Status CommonModel<T>::PythonPluginPreproces(std::vector<ForwardRequest>& forward_reqs) {
  size_t batch_size = forward_reqs.size();
  for (size_t idx = 0; idx < batch_size && forward_reqs[idx].step == 0; idx++) {
    py::gil_scoped_acquire acquire;

    auto ksana_python_input = std::make_shared<KsanaPythonInput>();
    ksana_python_input->input_tokens = *forward_reqs[idx].output_tokens;
    ksana_python_input->input_refit_embedding.pos = (*forward_reqs[idx].input_refit_embedding).pos;

    auto& embeddings = (*forward_reqs[idx].input_refit_embedding).embeddings;
    auto& tensors = ksana_python_input->input_refit_embedding.embedding_tensors;
    tensors.resize(embeddings.size());
    auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);

    // vector<vector<float>> to list[tensor]
    for (int i = 0; i < static_cast<int>(embeddings.size()); i++) {
      torch::Tensor input_refit_embedding_tensor =
          torch::from_blob(embeddings[i].data(), {static_cast<int64_t>(embeddings[i].size())}, options);
      tensors[i] =
          pybind11::reinterpret_borrow<pybind11::object>(py::handle(THPVariable_Wrap(input_refit_embedding_tensor)));
    }

    py::dict kwargs;
    kwargs["ksana_python_input"] = ksana_python_input;
    plugin_->attr("preprocess")(**kwargs);

    // list[tensor] to vector<vector<float>>
    embeddings.resize(tensors.size());
    for (int i = 0; i < static_cast<int>(embeddings.size()); i++) {
      py::object value_obj = py::reinterpret_borrow<py::object>(tensors[i]);
      torch::Tensor input_refit_embedding_tensor = THPVariable_Unpack(value_obj.ptr());
      int64_t output_number = input_refit_embedding_tensor.numel();
      embeddings[i].resize(output_number);
      memcpy(embeddings[i].data(), input_refit_embedding_tensor.data_ptr(), sizeof(float) * output_number);
    }
    // list[int] to vector<int>
    (*forward_reqs[idx].input_refit_embedding).pos = std::move(ksana_python_input->input_refit_embedding.pos);
  }
  return Status();
}

template <typename T>
Status CommonModel<T>::ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                     std::vector<ForwardRequest>& forward_reqs) {
  if (plugin_ && rank_ == 0) {
    PythonPluginPreproces(forward_reqs);
  }
  return CommonForward(base_weight, forward_reqs);
}

template <typename T>
Status CommonModel<T>::Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                              std::vector<ForwardRequest>& forward_reqs) {
  // TODO(zakwang): Remove ContextDecode
  if (plugin_ && rank_ == 0) {
    PythonPluginPreproces(forward_reqs);
  }
  return CommonForward(base_weight, forward_reqs);
}

template class CommonModel<float>;
template class CommonModel<float16>;
#ifdef ENABLE_BFLOAT16
template class CommonModel<bfloat16>;
#endif

}  // namespace ksana_llm
