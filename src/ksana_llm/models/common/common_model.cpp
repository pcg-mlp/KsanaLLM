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

  num_layer_ = model_config_.num_layer;
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

  bool is_neox = model_run_config.is_neox;
  bool is_alibi = (model_run_config.position_encoding == PositionEncoding::ALIBI);
  BlockManagerConfig block_manager_config;
  STATUS_CHECK_FAILURE(Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config));

  size_t max_token_num = model_config_.max_scheduler_token_num;
  qkv_add_bias_ = model_run_config.qkv_add_bias;
  KLLM_LOG_DEBUG << fmt::format("Max_Batch_Size = {}, Max Seq Len = {}, Max Token Num = {}",
                                model_config_.max_batch_size, model_config_.max_token_num, max_token_num);

  int inter_size_per_tp = model_config_.inter_size / tensor_para_size;
  int max_dim =
      std::max(std::max((head_num_per_tp + 2 * num_kv_heads_per_tp) * size_per_head, hidden_units), inter_size_per_tp);
  size_t tensor_buffer_size = std::max(model_config_.max_batch_size * vocab_size_pad_, max_token_num * max_dim);
  size_t up_matmul_tensor_buffer_size = max_token_num * std::max(static_cast<int>(inter_size_per_tp), hidden_units * 2);

  DataType weight_type = model_config_.weight_data_type;
  // TODO(karlluo): all create tensor used dynamic memory pool
  STATUS_CHECK_FAILURE(CreateBufferTensor(tensor_buffer_0_, {tensor_buffer_size}, weight_type));
  STATUS_CHECK_FAILURE(CreateBufferTensor(tensor_buffer_1_, {tensor_buffer_size}, weight_type));
  STATUS_CHECK_FAILURE(CreateBufferTensor(tensor_buffer_2_, {max_token_num, (size_t)max_dim}, weight_type));
  STATUS_CHECK_FAILURE(CreateBufferTensor(up_matmul_tensor_buffer_, {up_matmul_tensor_buffer_size}, weight_type));
  STATUS_CHECK_FAILURE(CreateBufferTensor(cos_sin_cache_tensor_,
                                          {(size_t)rotary_embedding, (size_t)max_position_embeddings}, weight_type));
#ifdef ENABLE_ACL
  STATUS_CHECK_FAILURE(CreateBufferTensor(ascend_buffer_0_, {max_token_num, hidden_units}, TYPE_FP16));
  STATUS_CHECK_FAILURE(CreateBufferTensor(ascend_buffer_1_, {max_token_num, hidden_units}, TYPE_FP16));
  STATUS_CHECK_FAILURE(CreateBufferTensor(ascend_buffer_2_, {max_token_num, hidden_units}, TYPE_FP16));
  STATUS_CHECK_FAILURE(CreateBufferTensor(ascend_buffer_3_, {max_token_num, hidden_units}, TYPE_FP16));
  STATUS_CHECK_FAILURE(CreateBufferTensor(ascend_buffer_4_, {max_token_num, hidden_units}, TYPE_FP16));

  for (int idx = 0; idx < num_layer_; ++idx) {
    Tensor key_cache;
    Tensor val_cache;
    STATUS_CHECK_FAILURE(CreateBufferTensor(key_cache, {max_token_num, hidden_units}, TYPE_FP16));
    STATUS_CHECK_FAILURE(CreateBufferTensor(val_cache, {max_token_num, hidden_units}, TYPE_FP16));

    ascend_key_caches_.push_back(key_cache);
    ascend_val_caches_.push_back(val_cache);
  }

#endif
  // TODO(karlluo): we needn't tensor's shape to transfer attribute
  STATUS_CHECK_FAILURE(CreateBufferTensor(forward_shape_, {1}, TYPE_INT32));

  KLLM_LOG_DEBUG << "Total buffer tensors memory used: " << (GetBufferTensorsMemoryUsed() >> 20) << " MB";

  // Initialize instances for each layer.
  emb_lookup_layer_ = std::make_shared<EmbLookupLayer<T>>();
  emb_lookup_layer_->Init({}, context_, rank_);

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
  model_output_ = std::make_shared<ModelOutput>(model_config_.max_batch_size, vocab_size_pad_, rank_, context_);
  model_communicator_ = std::make_shared<ModelCommunicator<T>>(&tensor_buffer_0_, &tensor_buffer_2_, rank_, context_);

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
#ifdef ENABLE_ACL
    attention_param.push_back(num_layer_);
#endif
    attention_param.push_back(max_position_embeddings);
    attention_param.push_back(head_num_per_tp);
    attention_param.push_back(num_kv_heads_per_tp);
    attention_param.push_back(size_per_head);
    attention_param.push_back(stride_size);
    attention_param.push_back(tensor_para_size);
    attention_param.push_back(GetBlockManager()->GetDtype());
    attention_param.push_back(rotary_embedding);
    attention_param.push_back(rope_theta);
    attention_param.push_back(is_neox);
    attention_param.push_back(is_alibi);
    attention_param.push_back(std::any(cos_sin_cache_ptr));
    attention_param.push_back(model_config_.rope_scaling_factor_config);

    flash_attention_layers_[idx]->Init(attention_param, context_, rank_);
    paged_attention_layers_[idx]->Init(attention_param, context_, rank_);
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
  auto matmul_layer_factory = MatMulLayerFactory<T>();
  attn_qkv_proj_layer_ =
      matmul_layer_factory.AutoCreateLayer(base_weight, "model.layers.0.self_attn.query_key_value.weight", weight_type,
                                           input_type, output_type, model_config_, {}, context_, rank_);
  attn_o_proj_layer_ =
      matmul_layer_factory.AutoCreateLayer(base_weight, "model.layers.0.self_attn.o_proj.weight", weight_type,
                                           input_type, output_type, model_config_, {}, context_, rank_);
  mlp_gate_proj_layer_ =
      matmul_layer_factory.AutoCreateLayer(base_weight, "model.layers.0.mlp.gate_proj.weight", weight_type, input_type,
                                           output_type, model_config_, {}, context_, rank_);
  mlp_up_proj_layer_ =
      matmul_layer_factory.AutoCreateLayer(base_weight, "model.layers.0.mlp.up_proj.weight", weight_type, input_type,
                                           output_type, model_config_, {}, context_, rank_);
  mlp_down_proj_layer_ =
      matmul_layer_factory.AutoCreateLayer(base_weight, "model.layers.0.mlp.down_proj.weight", weight_type, input_type,
                                           output_type, model_config_, {}, context_, rank_);
  lm_head_proj_layer_ = matmul_layer_factory.AutoCreateLayer(base_weight, "lm_head.weight", weight_type, input_type,
                                                             output_type, model_config_, {}, context_, rank_);

  // get maximum matmul workspace size and malloc workspace buffer
  std::vector<size_t> each_size = {
      attn_qkv_proj_layer_->GetWorkSpaceSize(), attn_o_proj_layer_->GetWorkSpaceSize(),
      mlp_gate_proj_layer_->GetWorkSpaceSize(), mlp_up_proj_layer_->GetWorkSpaceSize(),
      mlp_down_proj_layer_->GetWorkSpaceSize(), lm_head_proj_layer_->GetWorkSpaceSize(),
  };
  size_t shared_matmul_workspace_buffer_size = *std::max_element(each_size.begin(), each_size.end());
  if (shared_matmul_workspace_buffer_size > 0) {
    STATUS_CHECK_FAILURE(CreateBufferTensor(shared_matmul_workspace_buffer_,
                                            {static_cast<size_t>(shared_matmul_workspace_buffer_size)}, TYPE_UINT8));
  }

#ifdef ENABLE_ACL
  attn_qkv_proj_layer_->Init({}, context_, rank_);
  attn_o_proj_layer_->Init({}, context_, rank_);
  mlp_gate_proj_layer_->Init({}, context_, rank_);
  mlp_up_proj_layer_->Init({}, context_, rank_);
  mlp_down_proj_layer_->Init({}, context_, rank_);
  lm_head_proj_layer_->Init({}, context_, rank_);
#endif

  // set matumul workspace buffer
  attn_qkv_proj_layer_->SetWorkSpaceBuffer(shared_matmul_workspace_buffer_);
  attn_o_proj_layer_->SetWorkSpaceBuffer(shared_matmul_workspace_buffer_);
  mlp_gate_proj_layer_->SetWorkSpaceBuffer(shared_matmul_workspace_buffer_);
  mlp_up_proj_layer_->SetWorkSpaceBuffer(shared_matmul_workspace_buffer_);
  mlp_down_proj_layer_->SetWorkSpaceBuffer(shared_matmul_workspace_buffer_);
  lm_head_proj_layer_->SetWorkSpaceBuffer(shared_matmul_workspace_buffer_);

  // matmul layer preprocess
  attn_qkv_proj_layer_->Preprocess(model_config_);
  attn_o_proj_layer_->Preprocess(model_config_);
  mlp_gate_proj_layer_->Preprocess(model_config_);
  mlp_up_proj_layer_->Preprocess(model_config_);
  mlp_down_proj_layer_->Preprocess(model_config_);
  lm_head_proj_layer_->Preprocess(model_config_);

  return Status();
}

template <typename T>
Status CommonModel<T>::AddPrefixCache(std::vector<Tensor>& mmha_origin_input, std::vector<Tensor>& mmha_prefix_input) {
  // Before MMHA inference, retrieve the key-value pairs (k, v) from the Prefix Cache Block
  // and populate them into the mmha prefix input tensor.
  size_t total_token_num = 0;
  size_t dtype_size = GetTypeSize(mmha_origin_input[0].dtype);
  size_t size_per_token = mmha_origin_input[0].shape[1] * dtype_size;
  for (size_t idx = 0; idx < model_input_->batch_size; ++idx) {
    size_t src_offset = (model_input_->input_offset_list[idx] - model_input_->input_prefix_list[idx]) * size_per_token;
    size_t input_token_num = model_input_->input_offset_list[idx + 1] - model_input_->input_offset_list[idx];
    size_t prefix_token_num = model_input_->input_prefix_list[idx + 1] - model_input_->input_prefix_list[idx];
    size_t copy_size = (input_token_num - prefix_token_num) * size_per_token;
    size_t dst_offset = (model_input_->input_offset_list[idx] + prefix_token_num) * size_per_token;
    MemcpyAsync(mmha_prefix_input[0].GetPtr<void>() + dst_offset, mmha_origin_input[0].GetPtr<void>() + src_offset,
                copy_size, MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
    total_token_num += input_token_num;
  }
  mmha_prefix_input[0].shape = {total_token_num, mmha_origin_input[0].shape[1]};
  mmha_prefix_input[0].dtype = mmha_origin_input[0].dtype;
  StreamSynchronize(context_->GetComputeStreams()[rank_]);
  return Status();
}

template <typename T>
Status CommonModel<T>::RemovePrefixCache(std::vector<Tensor>& mmha_prefix_output, std::vector<Tensor>& mmha_output) {
  // After the completion of MMHA inference, copy the data from the MMHA output results,
  // excluding the Prefix Cache section, and continue with the subsequent inference.
  size_t dst_offset = 0;
  size_t src_offset = 0;
  size_t dtype_size = GetTypeSize(mmha_prefix_output[0].dtype);
  size_t size_per_token = mmha_prefix_output[0].shape[1] * dtype_size;
  for (size_t idx = 0; idx < model_input_->batch_size; ++idx) {
    size_t prefix_length = model_input_->input_prefix_list[idx + 1] - model_input_->input_prefix_list[idx];
    size_t input_length = model_input_->input_offset_list[idx + 1] - model_input_->input_offset_list[idx];
    src_offset += prefix_length * size_per_token;
    size_t copy_size = size_per_token * (input_length - prefix_length);

    MemcpyAsync(mmha_output[0].GetPtr<void>() + dst_offset, mmha_prefix_output[0].GetPtr<void>() + src_offset,
                copy_size, MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
    src_offset += copy_size;
    dst_offset += copy_size;
  }
  StreamSynchronize(context_->GetComputeStreams()[rank_]);
  return Status();
}

template <typename T>
bool CommonModel<T>::IsPrefixCachingComputationReuse() {
#ifdef ENABLE_ACL
  // NPU device does not currently support prefix caching for computation reuse.
  return false;
#endif
  return GetBlockManager()->GetPrefixCacheBlocksNumber() > 0;
}

template <typename T>
Status CommonModel<T>::FlashAttentionForward(std::vector<Tensor>& flash_attention_input,
                                             std::vector<Tensor>& flash_attention_output, int layer_idx) {
  STATUS_CHECK_RETURN(flash_attention_layers_[layer_idx]->Forward(
      {flash_attention_input[0], model_input_->input_offset_uint64_tensor, model_input_->kv_list,
       model_input_->input_prefix_uint64_tensor, model_input_->kv_cache_offset_tensor,
       model_input_->rotary_embedding_pos, model_input_->rotary_embedding_mask, forward_shape_
#ifdef ENABLE_ACL
       ,
       ascend_buffer_0_, ascend_buffer_1_, ascend_buffer_2_, ascend_buffer_3_, ascend_buffer_4_,
       ascend_key_caches_[layer_idx], ascend_val_caches_[layer_idx]
#endif
      },
      flash_attention_output));
  return Status();
}

template <typename T>
Status CommonModel<T>::CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                       Tensor& hidden_states, std::vector<Tensor>& temp_buffer_0,
                                       std::vector<Tensor>& temp_buffer_1, std::vector<Tensor>& temp_buffer_2,
                                       const bool is_context_stage) {
  // Attn proj MatMul
  Tensor attn_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.query_key_value.weight", layer_idx));
  std::vector<Tensor>& attn_proj_output = temp_buffer_2;
  STATUS_CHECK_RETURN(attn_qkv_proj_layer_->Forward({hidden_states, attn_proj_weight}, attn_proj_output));
  if (qkv_add_bias_) {
    Tensor attn_proj_bias =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.query_key_value.bias", layer_idx));
    STATUS_CHECK_RETURN(add_layer_->Forward({attn_proj_output[0], attn_proj_bias}, attn_proj_output));
  }

  // MMHA Flash/Paged Attention
  std::vector<Tensor>& mmha_attention_output = temp_buffer_1;
  if (layer_idx == 0) {
    // only need sync in the first layer
    StreamWaitEvent(context_->GetComputeStreams()[rank_], model_input_->kvcache_offset_event);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], model_input_->rotary_embedding_event);
  }

  if (is_context_stage) {
    if (IsPrefixCachingComputationReuse()) {
      size_t input_tokens_num = attn_proj_output[0].shape[0];
      std::vector<Tensor>& mmha_prefix_input = temp_buffer_1;
      AddPrefixCache(attn_proj_output, mmha_prefix_input);

      std::vector<Tensor>& mmha_prefix_output = temp_buffer_2;
      FlashAttentionForward(mmha_prefix_input, mmha_prefix_output, layer_idx);

      RemovePrefixCache(mmha_prefix_output, mmha_attention_output);
      mmha_attention_output[0].shape[0] = input_tokens_num;
      mmha_attention_output[0].shape[1] = mmha_prefix_output[0].shape[1];
    } else {
      FlashAttentionForward(attn_proj_output, mmha_attention_output, layer_idx);
    }
  } else {
    STATUS_CHECK_RETURN(paged_attention_layers_[layer_idx]->Forward(
        {attn_proj_output[0], model_input_->input_tokens_int32_tensor, model_input_->kv_list,
         model_input_->kv_cache_offset_tensor, model_input_->rotary_embedding_pos, model_input_->rotary_embedding_mask,
         model_input_->kv_cache_buffer, forward_shape_, up_matmul_tensor_buffer_
#ifdef ENABLE_ACL
         ,
         ascend_buffer_0_, ascend_buffer_1_, ascend_buffer_2_, ascend_buffer_3_, ascend_buffer_4_,
         ascend_key_caches_[layer_idx], ascend_val_caches_[layer_idx]
#endif
        },
        mmha_attention_output));
  }

  // Attn o_proj MatMul
  Tensor attn_o_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx));
  std::vector<Tensor>& attn_o_proj_output = temp_buffer_2;
  STATUS_CHECK_RETURN(attn_o_proj_layer_->Forward({mmha_attention_output[0], attn_o_proj_weight}, attn_o_proj_output));

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }
  // Attn NcclAllReduceSum
  std::vector<Tensor>& attn_all_reduce_sum_output = temp_buffer_1;
  model_communicator_->ReduceSum(attn_o_proj_output, attn_all_reduce_sum_output, is_context_stage, true);

  return Status();
}

template <typename T>
Status CommonModel<T>::CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                 Tensor& post_layernorm_output, std::vector<Tensor>& temp_buffer_0,
                                 std::vector<Tensor>& temp_buffer_1, std::vector<Tensor>& temp_buffer_2) {
  // Mlp gate_proj MatMul
  Tensor gate_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.gate_proj.weight", layer_idx));
  std::vector<Tensor>& gate_matmul_output = temp_buffer_0;
  STATUS_CHECK_RETURN(mlp_gate_proj_layer_->Forward({post_layernorm_output, gate_proj_weight}, gate_matmul_output));

  // Mlp up_proj MatMul 由于 gate_proj 与 up_proj 为并行关系,因此此处使用额外空间存储 matmul 结果
  Tensor up_proj_weight = base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.up_proj.weight", layer_idx));
  std::vector<Tensor> up_matmul_output = {up_matmul_tensor_buffer_};
  STATUS_CHECK_RETURN(mlp_up_proj_layer_->Forward({post_layernorm_output, up_proj_weight}, up_matmul_output));

  std::vector<Tensor>& silu_mul_output = temp_buffer_1;
  STATUS_CHECK_RETURN(silu_mul_layer_->Forward({gate_matmul_output[0], up_matmul_output[0]}, silu_mul_output));

  // Mlp down_proj MatMul
  Tensor down_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.down_proj.weight", layer_idx));
  std::vector<Tensor>& down_proj_output = temp_buffer_0;
  STATUS_CHECK_RETURN(mlp_down_proj_layer_->Forward({silu_mul_output[0], down_proj_weight}, down_proj_output));

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }
  // Mlp NcclAllReduceSum
  std::vector<Tensor>& mlp_all_reduce_sum_output = temp_buffer_1;
  model_communicator_->ReduceSum({down_proj_output[0]}, mlp_all_reduce_sum_output, false, false);

  return Status();
}

template <typename T>
Status CommonModel<T>::CommonDecoder(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                     std::vector<Tensor>& temp_buffer_0, std::vector<Tensor>& temp_buffer_1,
                                     std::vector<Tensor>& temp_buffer_2, const bool is_context_stage) {
  // input layernorm
  Tensor input_layernorm_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.input_layernorm.weight", layer_idx));
  // input_layernorm_input = layer_idx == 0 ? emb_lookup_output : mlp_add_output
  // Since emb_lookup_output and mlp_add_output point to the same memory address, we implement it as follow:
  std::vector<Tensor>& input_layernorm_input = temp_buffer_0;
  std::vector<Tensor>& input_layernorm_output = temp_buffer_1;

  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({input_layernorm_input[0], input_layernorm_weight}, input_layernorm_output));

  STATUS_CHECK_RETURN(CommonAttention(layer_idx, base_weight, input_layernorm_output[0], temp_buffer_0, temp_buffer_1,
                                      temp_buffer_2, is_context_stage));

  // Attn Add
  std::vector<Tensor>& attn_all_reduce_sum_output = temp_buffer_1;
  std::vector<Tensor>& attn_add_output = temp_buffer_2;
  STATUS_CHECK_RETURN(add_layer_->Forward({input_layernorm_input[0], attn_all_reduce_sum_output[0]}, attn_add_output));

  // post_attention_layernorm
  Tensor post_layernorm_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.post_attention_layernorm.weight", layer_idx));
  std::vector<Tensor>& post_layernorm_output = temp_buffer_1;
  STATUS_CHECK_RETURN(layernorm_layer_->Forward({attn_add_output[0], post_layernorm_weight}, post_layernorm_output));

  STATUS_CHECK_RETURN(
      CommonMlp(layer_idx, base_weight, post_layernorm_output[0], temp_buffer_0, temp_buffer_1, temp_buffer_2));

  // Mlp Add
  std::vector<Tensor>& mlp_all_reduce_sum_output = temp_buffer_1;
  std::vector<Tensor>& mlp_add_output = temp_buffer_0;
  STATUS_CHECK_RETURN(add_layer_->Forward({mlp_all_reduce_sum_output[0], attn_add_output[0]}, mlp_add_output));
  return Status();
}

template <typename T>
Status CommonModel<T>::EmbedTokensUseCpu(Tensor& embedding_weight, std::vector<ForwardRequest>& forward_reqs,
                                         const bool is_context_stage, std::vector<Tensor>& temp_buffer_0) {
  auto batch_size = forward_reqs.size();
  void* input_tokens_ptr = cpu_input_tokens_tensor_.GetPtr<void>();
  size_t index = 0;
  for (size_t idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    if (is_context_stage) {
      size_t copy_len = req_input->size() * sizeof(int);
      memcpy(input_tokens_ptr + index, req_input->data(), copy_len);
      index += copy_len;
    } else {
      memcpy(input_tokens_ptr + index, &req_input->back(), sizeof(int));
      index += sizeof(int);
    }
  }
  cpu_input_tokens_tensor_.shape = {index / sizeof(int)};
  cpu_emb_lookup_layer_->Forward({cpu_input_tokens_tensor_, cpu_tokens_emb_tensor_, embedding_weight}, temp_buffer_0);
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
    std::vector<std::pair<size_t, size_t>> slice_pos = it->second.slice_pos;
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
                                     std::vector<ForwardRequest>& forward_reqs, const bool is_context_stage) {
  GetBlockManager()->SetDeviceId(rank_);

  // 推理前准备三块循环使用的推理时临时空间, 用于暂存各层输出结果
  std::vector<Tensor> temp_buffer_0{tensor_buffer_0_};
  std::vector<Tensor> temp_buffer_1{tensor_buffer_1_};
  std::vector<Tensor> temp_buffer_2{tensor_buffer_2_};
  Tensor embedding_weight = base_weight->GetModelWeights("model.embed_tokens.weight");
  if (embedding_weight.device == MemoryDevice::MEMORY_HOST) {
    EmbedTokensUseCpu(embedding_weight, forward_reqs, is_context_stage, temp_buffer_0);
  }

  model_input_->ParseFromRequests(forward_reqs, is_context_stage);

  // create forward shape tensor
  forward_shape_.shape = {model_input_->batch_size, model_input_->max_tokens,
                          (size_t)model_input_->kv_cache_offset_list.back()};

  std::vector<Tensor>& emb_lookup_output = temp_buffer_0;
  StreamWaitEvent(context_->GetComputeStreams()[rank_], model_input_->input_ids_event);
  if (embedding_weight.device == MemoryDevice::MEMORY_DEVICE) {
    STATUS_CHECK_RETURN(emb_lookup_layer_->Forward({model_input_->input_ids, model_input_->input_offset_uint64_tensor,
                                                    model_input_->input_prefix_uint64_tensor, embedding_weight},
                                                   emb_lookup_output));

    // NOTE(karlluo): multiple event in nccl will cause preformance regression
    // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
      StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
    }

    model_communicator_->AllGather({emb_lookup_output[0], temp_buffer_1[0]}, emb_lookup_output);
  }

  // refit input needs to be processed only in the context stage.
  if (is_context_stage) {
    input_refit_layer_->Forward({model_input_->cpu_input_refit_tensor.pos_pair_tensor,
                                 model_input_->cpu_input_refit_tensor.emb_fp32_ptr_tensor},
                                emb_lookup_output);
  }

  // CommonDecoder
  for (int layer_idx = 0; layer_idx < num_layer_; ++layer_idx) {
    STATUS_CHECK_RETURN(
        CommonDecoder(layer_idx, base_weight, temp_buffer_0, temp_buffer_1, temp_buffer_2, is_context_stage));
  }

  if (is_context_stage) {
    if (UpdateResponse(forward_reqs, temp_buffer_0[0], "transformer")) {
      StreamSynchronize(context_->GetComputeStreams()[rank_]);
      input_refit_layer_->Clear();
      return Status();
    }
  }

  // final norm
  Tensor final_layernorm_weight = base_weight->GetModelWeights("model.norm.weight");
  std::vector<Tensor>& final_layernorm_input = temp_buffer_0;
  std::vector<Tensor>& final_layernorm_output = temp_buffer_1;
  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({final_layernorm_input[0], final_layernorm_weight}, final_layernorm_output));

  if (is_context_stage) {
    if (UpdateResponse(forward_reqs, final_layernorm_output[0], "layernorm")) {
      StreamSynchronize(context_->GetComputeStreams()[rank_]);
      input_refit_layer_->Clear();
      return Status();
    }
  }

  // assemble last token
  std::vector<Tensor>& assemble_last_token_output = temp_buffer_2;
  if (model_input_->use_logits_custom_length) {
    STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward(
        {final_layernorm_output[0], model_input_->logits_custom_length_uint64_tensor,
         model_input_->logits_length_prefix_uint64_tensor},
        assemble_last_token_output));
  } else {
    STATUS_CHECK_RETURN(assemble_last_token_layer_->Forward(
        {final_layernorm_output[0], model_input_->input_offset_uint64_tensor, model_input_->input_prefix_uint64_tensor},
        assemble_last_token_output));
  }

  // lm_head
  Tensor lm_head_weight = base_weight->GetModelWeights("lm_head.weight");
  std::vector<Tensor>& lm_head_output = temp_buffer_0;
  STATUS_CHECK_RETURN(lm_head_proj_layer_->Forward({assemble_last_token_output[0], lm_head_weight}, lm_head_output));

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }

  model_communicator_->AllGather({lm_head_output[0], temp_buffer_1[0]}, lm_head_output);

  // Cast to float & Copy to logits buffer
  forward_shape_.shape = {forward_reqs[0].logits_offset * vocab_size_pad_ * sizeof(float)};
  std::vector<Tensor> logits_buffer{model_output_->logits_tensor};
  STATUS_CHECK_RETURN(cast_layer_->Forward({lm_head_output[0], forward_shape_}, logits_buffer));
  StreamSynchronize(context_->GetComputeStreams()[rank_]);
  input_refit_layer_->Clear();
  return Status();
}

template <typename T>
Status CommonModel<T>::PythonPluginPreproces(std::vector<ForwardRequest>& forward_reqs) {
  size_t batch_size = forward_reqs.size();
  for (size_t idx = 0; idx < batch_size; idx++) {
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
  return CommonForward(base_weight, forward_reqs, /*is_context_stage*/ true);
}

template <typename T>
Status CommonModel<T>::Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                              std::vector<ForwardRequest>& forward_reqs) {
  return CommonForward(base_weight, forward_reqs, /*is_context_stage*/ false);
}

template class CommonModel<float>;
template class CommonModel<float16>;
#ifdef ENABLE_BFLOAT16
template class CommonModel<bfloat16>;
#endif

}  // namespace ksana_llm
